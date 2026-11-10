"""One training run = one dict in, one dict out.

    run({'name': 'baseline', 'mode': 'standard', 'batch': 32, 'tokens': 50e6, ...})

Everything that differs between the runs in this repo is a key in that dict.  The returned
dict holds the loss curves (by tokens seen, so different batch sizes share an x-axis), the
throughput, the peak memory, the wall time and the dollar cost.
"""
from __future__ import annotations

import gc
import math
import os
import time

import numpy as np
import torch

from .data import Windows, data_dir
from .model import GPT, ModelCfg
from .reversible import RevCfg

# Modal's list price for an A100-40GB (USD per GPU-hour), used only to turn seconds into dollars.
PRICE_PER_HOUR = {'A100-SXM4-40GB': 2.10, 'A100-SXM4-80GB': 2.50, 'NVIDIA H100 80GB HBM3': 3.95,
                  'Tesla T4': 0.59, 'NVIDIA L4': 0.80, 'cpu': 0.0}

DEFAULTS = dict(
    name='run', mode='standard', integrator='midpoint', h=0.5, a=0.5, bootstrap='euler', state_dtype='float32', anchor=False,
    n_layer=12, width=384, n_head=6, seq=1024, batch=32, tokens=50_000_000,
    lr=2e-3, warmup_frac=0.05, min_lr_frac=0.1, wd=0.0, beta1=0.9, beta2=0.95, clip=1.0, seed=0,
    ce_chunk=0, eval_every=5_000_000, eval_tokens=524_288, final_eval_tokens=2_097_152, eval_batch=16,
    save='', samples=True, warmup_steps_timing=5,
)


_CAPPED = False


def device():
    """The training device. With S13_MEM_CAP_GIB set, the CUDA allocator is capped at that many GiB, so every
    memory limit in this repo is the limit of a 40GB A100 even if a bigger card is underneath."""
    global _CAPPED
    if torch.cuda.is_available():
        cap = os.environ.get('S13_MEM_CAP_GIB')
        if cap and not _CAPPED:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * 2**30 / total))
            _CAPPED = True
        return torch.device('cuda')
    return torch.device('cpu')


def amp_dtype(dev):
    if dev.type != 'cuda':
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def gpu_name(dev):
    return torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'


def full_cfg(cfg):
    c = dict(DEFAULTS)
    c.update(cfg)
    return c


def build(c, dev):
    rc = RevCfg(c['integrator'], c['h'], c['a'], c['bootstrap'], c['state_dtype'], c.get('anchor', False))
    mc = ModelCfg(n_layer=c['n_layer'], width=c['width'], n_head=c['n_head'], mode=c['mode'], rev=rc,
                  ce_chunk=c['ce_chunk'])
    torch.manual_seed(c['seed'])
    return GPT(mc).to(dev)


def make_opt(model, c, dev):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    groups = [{'params': decay, 'weight_decay': c['wd']}, {'params': other, 'weight_decay': 0.0}]
    return torch.optim.AdamW(groups, lr=c['lr'], betas=(c['beta1'], c['beta2']), eps=1e-8,
                             fused=dev.type == 'cuda')


def lr_at(c, frac):
    w = c['warmup_frac']
    if frac < w:
        return c['lr'] * (frac + 1e-9) / w
    p = min(1.0, (frac - w) / max(1e-9, 1 - w))
    m = c['min_lr_frac']
    return c['lr'] * (m + (1 - m) * 0.5 * (1 + math.cos(math.pi * p)))


def autocast(dev):
    dt = amp_dtype(dev)
    return torch.autocast(dev.type, dtype=dt, enabled=dt is not None)


@torch.no_grad()
def evaluate(model, val, c, dev, n_tokens):
    model.eval()
    T, B = c['seq'], c['eval_batch']
    n_seq = max(B, n_tokens // T)
    tot, k = 0.0, 0
    for o in range(0, n_seq, B):
        x, y = val.get(o, min(B, n_seq - o), dev)
        with autocast(dev):
            tot += model(x, y).item() * x.size(0)
        k += x.size(0)
    model.train()
    return tot / k


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(cfg: dict, verbose: bool = True) -> dict:
    c = full_cfg(cfg)
    dev = device()
    dd = data_dir()
    train = Windows(dd / 'train.bin', c['seq'], seed=1234)
    val = Windows(dd / 'val.bin', c['seq'], seed=99)
    model = build(c, dev)
    opt = make_opt(model, c, dev)
    scaler = torch.amp.GradScaler(enabled=amp_dtype(dev) == torch.float16)
    B, T = c['batch'], c['seq']
    tok_step = B * T
    steps = max(1, int(c['tokens'] // tok_step))
    log_every = max(1, min(10, steps // 200))
    if dev.type == 'cuda':
        torch.cuda.synchronize()
        free()
        torch.cuda.reset_peak_memory_stats()
    static_mem = torch.cuda.memory_allocated() if dev.type == 'cuda' else 0

    curve = {'tokens': [], 'loss': [], 'lr': [], 'gnorm': []}
    evals = {'tokens': [], 'val': []}
    next_eval = c['eval_every']
    loss_acc = torch.zeros((), device=dev)
    gn_acc = torch.zeros((), device=dev)
    train_time, timed_tokens, diverged = 0.0, 0, False
    t_start = time.time()
    if verbose:
        print(f"[{c['name']}] {steps} steps x {tok_step:,} tok | mode={c['mode']} "
              f"{c['integrator'] if c['mode'].startswith('rev') else ''} | {gpu_name(dev)}", flush=True)

    for step in range(steps):
        frac = step / steps
        lr = lr_at(c, frac)
        for g in opt.param_groups:
            g['lr'] = lr
        x, y = train.get(step * B, B, dev)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast(dev):
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c['clip'])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= c['warmup_steps_timing'] or steps <= c['warmup_steps_timing'] * 2:
            train_time += dt
            timed_tokens += tok_step
        loss_acc += loss.detach()
        gn_acc += gn.detach()
        if (step + 1) % log_every == 0 or step == steps - 1:
            n = (step % log_every) + 1
            lv = (loss_acc / n).item()
            curve['tokens'].append((step + 1) * tok_step)
            curve['loss'].append(round(lv, 5))
            curve['lr'].append(lr)
            curve['gnorm'].append(round((gn_acc / n).item(), 4))
            loss_acc.zero_(); gn_acc.zero_()
            if not math.isfinite(lv):
                diverged = True
                break
        if (step + 1) * tok_step >= next_eval and step < steps - 1:
            evals['tokens'].append((step + 1) * tok_step)
            evals['val'].append(round(evaluate(model, val, c, dev, c['eval_tokens']), 5))
            next_eval += c['eval_every']
            if verbose:
                print(f"[{c['name']}] {(step + 1) * tok_step / 1e6:6.1f}M tok  train {curve['loss'][-1]:.4f}  "
                      f"val {evals['val'][-1]:.4f}  {timed_tokens / max(train_time, 1e-9):,.0f} tok/s", flush=True)

    peak = torch.cuda.max_memory_allocated() if dev.type == 'cuda' else 0
    peak_reserved = torch.cuda.max_memory_reserved() if dev.type == 'cuda' else 0
    final_val = float('nan') if diverged else evaluate(model, val, c, dev, c['final_eval_tokens'])
    evals['tokens'].append(steps * tok_step)
    evals['val'].append(round(final_val, 5))
    wall = time.time() - t_start
    gname = gpu_name(dev)
    out = {
        'cfg': c, 'gpu': gname, 'torch': torch.__version__, 'amp': str(amp_dtype(dev)),
        'params_non_embedding': model.n_params(True), 'params_total': model.n_params(False),
        'steps': steps, 'tokens_per_step': tok_step, 'tokens_trained': steps * tok_step,
        'final_train': float(np.mean(curve['loss'][-max(1, len(curve['loss']) // 20):])),
        'final_val': final_val, 'diverged': diverged,
        'tok_per_s': timed_tokens / max(train_time, 1e-9), 'train_time_s': train_time, 'wall_s': wall,
        'peak_mem_bytes': peak, 'peak_reserved_bytes': peak_reserved, 'static_mem_bytes': static_mem,
        'cost_usd': wall / 3600 * PRICE_PER_HOUR.get(gname, 2.10),
        'curve': curve, 'evals': evals,
    }
    if c['samples'] and not diverged:
        out['samples'] = samples(model, dev)
    if c['save']:
        os.makedirs(os.path.dirname(c['save']), exist_ok=True)
        torch.save({'cfg': c, 'state': model.state_dict()}, c['save'])
    if verbose:
        print(f"[{c['name']}] done: val {final_val:.4f} | {out['tok_per_s']:,.0f} tok/s | "
              f"peak {peak / 2**30:.2f} GiB | {wall:.0f}s | ${out['cost_usd']:.3f}", flush=True)
    del model, opt
    free()
    return out


def samples(model, dev, prompts=('The most important thing about learning is', 'In the year 1850,'), n=48):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    torch.manual_seed(0)
    res = []
    for p in prompts:
        idx = torch.tensor([enc.encode(p)], device=dev)
        with autocast(dev):
            o = model.generate(idx, n)
        res.append(enc.decode(o[0].tolist()))
    return res
