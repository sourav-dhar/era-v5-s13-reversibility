"""A ~20M (non-embedding) GPT whose residual stack can run four ways.

    mode = 'standard'      x = x + f(x)                       autograd stores every activation
    mode = 'ckpt'          x = x + checkpoint(f, x)           store each block's input, recompute inside
    mode = 'rev_unrolled'  the reversible integrator, but with ordinary autograd (stores everything)
    mode = 'rev'           the reversible integrator with the custom backward (stores two states)

``f`` is the whole pre-norm transformer block minus its input, so 'standard' is exactly a
pre-norm GPT:  u = Attn(LN1 x);  f(x) = u + MLP(LN2(x + u)).

The LM head is tied to the embedding.  With a 50k vocabulary the logits are the largest single
activation in a model this narrow, so ``ce_chunk`` computes the loss over chunks of tokens with
the logits recomputed in backward.  That is the second memory lever studied in this repo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .reversible import RevCfg, integrate, rev_stack


@dataclass
class ModelCfg:
    vocab: int = 50304              # GPT-2's 50,257 padded to a multiple of 64
    n_layer: int = 12
    width: int = 384
    n_head: int = 6
    mlp_mult: int = 4
    mode: str = 'standard'          # 'standard' | 'ckpt' | 'rev_unrolled' | 'rev'
    rev: RevCfg = field(default_factory=RevCfg)
    ce_chunk: int = 0               # tokens per loss chunk; 0 = materialise every logit at once
    rope_base: float = 10000.0

    def asdict(self):
        d = asdict(self)
        d['rev'] = self.rev.asdict()
        return d


class Rotary(nn.Module):
    def __init__(s, head_dim, base):
        super().__init__()
        s.register_buffer('inv', 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim), persistent=False)
        s._cache = (0, None, None)

    def forward(s, T, device):
        if s._cache[0] < T or s._cache[1].device != device:
            t = torch.arange(max(T, 1024), device=device).float()
            fr = torch.outer(t, s.inv.to(device))
            s._cache = (len(t), fr.cos()[None, None], fr.sin()[None, None])
        return s._cache[1][:, :, :T], s._cache[2][:, :, :T]


def _rope(x, cos, sin):
    xf = x if x.dtype == torch.float64 else x.float()      # fp64 stays fp64 so the exactness tests are exact
    cos, sin = cos.to(xf.dtype), sin.to(xf.dtype)
    x1, x2 = xf.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).type_as(x)


class Block(nn.Module):
    """Returns f(x) = block(x) - x, the quantity every integrator in this repo consumes."""

    def __init__(s, c: ModelCfg, rotary: Rotary):
        super().__init__()
        C = c.width
        s.nh, s.hd = c.n_head, C // c.n_head
        s.ln1, s.ln2 = nn.LayerNorm(C, bias=False), nn.LayerNorm(C, bias=False)
        s.qkv, s.proj = nn.Linear(C, 3 * C, bias=False), nn.Linear(C, C, bias=False)
        s.fc, s.out = nn.Linear(C, c.mlp_mult * C, bias=False), nn.Linear(c.mlp_mult * C, C, bias=False)
        s.rotary = [rotary]          # shared, kept out of the module tree on purpose

    def attn(s, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, s.nh, s.hd).transpose(1, 2) for t in s.qkv(x).split(C, dim=2))
        cos, sin = s.rotary[0](T, x.device)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.proj(y.transpose(1, 2).reshape(B, T, C))

    def forward(s, x):
        u = s.attn(s.ln1(x))
        return u + s.out(F.gelu(s.fc(s.ln2(x + u))))


def _ce_sum(h, W, t):
    return F.cross_entropy((h @ W.t()).float(), t, reduction='sum')


class GPT(nn.Module):
    def __init__(s, c: ModelCfg):
        super().__init__()
        s.cfg = c
        s.wte = nn.Embedding(c.vocab, c.width)
        s.rotary = Rotary(c.width // c.n_head, c.rope_base)
        s.blocks = nn.ModuleList(Block(c, s.rotary) for _ in range(c.n_layer))
        s.ln_f = nn.LayerNorm(c.width, bias=False)
        s.apply(s._init)
        for b in s.blocks:  # GPT-2 residual-projection scaling
            for w in (b.proj.weight, b.out.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------ counting
    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.wte.weight.numel() if non_embedding else n

    # ------------------------------------------------------------------ trunk
    def trunk(s, x):
        m = s.cfg.mode
        if m == 'standard':
            for b in s.blocks:
                x = x + b(x)
            return x
        if m == 'ckpt':
            for b in s.blocks:
                x = x + checkpoint(b, x, use_reentrant=False)
            return x
        if m == 'rev_unrolled':
            return integrate(s.blocks, x, s.cfg.rev)
        if m == 'rev':
            return rev_stack(s.blocks, x, s.cfg.rev)
        raise ValueError(m)

    def hidden(s, idx):
        x = s.trunk(s.wte(idx))
        return s.ln_f(x.float() if x.dtype in (torch.bfloat16, torch.float16) else x)

    def forward(s, idx, targets=None):
        h = s.hidden(idx)
        W = s.wte.weight
        if targets is None:
            return h[:, -1:] @ W.t()
        if s.cfg.ce_chunk:
            hf, tf = h.reshape(-1, h.size(-1)), targets.reshape(-1)
            n, ch, tot = tf.numel(), s.cfg.ce_chunk, 0.0
            for i in range(0, n, ch):
                tot = tot + checkpoint(_ce_sum, hf[i:i + ch], W, tf[i:i + ch], use_reentrant=False)
            return tot / n
        logits = h @ W.t()
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))

    @torch.no_grad()
    def generate(s, idx, n_new, temperature=0.8, top_k=40):
        for _ in range(n_new):
            logits = s(idx[:, -1024:])[:, -1].float() / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx
