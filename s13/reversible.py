"""Reversible residual stacks (Gal, Eliasof, Turek, Ascher, Treister & Haber, 2025, arXiv:2512.02056).

Treat the layer index as time. Every integrator here is a *two-step* rule

    s[k+1] = alpha * s[k-1] + beta * s[k] + gamma * f_k(s[k])                      (1)

which can be read right-to-left because f_k is evaluated at s[k], the state the backward pass
already holds:

    s[k-1] = ( s[k+1] - beta * s[k] - gamma * f_k(s[k]) ) / alpha                  (2)

    integrator   alpha     beta     gamma     paper
    ----------   -----     ----     -----     -----
    midpoint       1         0        2h       Eq. 4   p+ = p- + 2h f(p)
    blend          a       1 - a       h       Eq. 15  the "a"-variant; Lightning LM: a = 0.5, h = 0.25
    leapfrog      -1         2        h^2      Eq. 6   p+ = 2p - p- + h^2 f(p)

A two-step rule needs two starting states. ``bootstrap`` decides the second one:
    'euler'  s[1] = s[0] + h f_0(s[0])       one ordinary (irreversible-looking, but s[0] is
                                              recovered by the recurrence anyway) Euler step
    'none'   s[1] = s[0]                      ("no_kick" in the Lightning LM library)

``f_k`` is the *whole* transformer block minus its input: attention followed by the MLP.

Two implementations of the same arithmetic live here:
  * ``integrate``  - plain PyTorch, autograd stores every activation.  This is the reference,
                     and also the "reversible architecture without the reversible backward" mode.
  * ``RevStack``   - a torch.autograd.Function that keeps only (s[L-1], s[L]) and, in backward,
                     walks down the stack rebuilding each state with (2).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass(frozen=True)
class RevCfg:
    integrator: str = 'midpoint'      # 'midpoint' | 'blend' | 'leapfrog'
    h: float = 0.5
    a: float = 0.5                    # only used by 'blend'
    bootstrap: str = 'euler'          # 'euler' | 'none'
    state_dtype: str = 'float32'      # dtype of the residual stream itself ('float32' | 'bfloat16')
    anchor: bool = False              # also keep s[0], s[1]: the bottom states are tiny next to the ones
                                      # above them, so rebuilding them by subtraction loses all precision

    def coeffs(self):
        if self.integrator == 'midpoint':
            return 1.0, 0.0, 2.0 * self.h
        if self.integrator == 'blend':
            return self.a, 1.0 - self.a, self.h
        if self.integrator == 'leapfrog':
            return -1.0, 2.0, self.h * self.h
        raise ValueError(self.integrator)

    @property
    def dtype(self):
        return getattr(torch, self.state_dtype)

    def label(self):
        if self.integrator == 'blend':
            core = f'blend a={self.a:g} h={self.h:g}'
        else:
            core = f'{self.integrator} h={self.h:g}'
        return f'{core}, boot={self.bootstrap}' + (', anchored' if self.anchor else '')

    def asdict(self):
        return asdict(self)


def _step(alpha, beta, gamma, prev, cur, fcur):
    new = gamma * fcur
    if beta:
        new = new + beta * cur
    return new + alpha * prev if alpha != 1.0 else new + prev


def _unstep(alpha, beta, gamma, new, cur, fcur):
    prev = new - gamma * fcur
    if beta:
        prev = prev - beta * cur
    return prev if alpha == 1.0 else prev / alpha


def integrate(blocks, s0: torch.Tensor, rc: RevCfg, keep_states: bool = False):
    """Run the reversible stack forward with ordinary autograd.  Returns s[L] (and every state)."""
    alpha, beta, gamma = rc.coeffs()
    dt = rc.dtype
    s0 = s0.to(dt)
    if rc.bootstrap == 'euler':
        s1, rest = s0 + rc.h * blocks[0](s0).to(dt), blocks[1:]
    else:
        s1, rest = s0, blocks
    states = [s0, s1]
    prev, cur = s0, s1
    for blk in rest:
        prev, cur = cur, _step(alpha, beta, gamma, prev, cur, blk(cur).to(dt))
        if keep_states:
            states.append(cur)
    return (cur, states) if keep_states else cur


@torch.no_grad()
def reconstruct(blocks, top_pair, rc: RevCfg):
    """Walk (s[L-1], s[L]) back down to s[0] with Eq. (2). Returns the rebuilt states, s[0] first."""
    alpha, beta, gamma = rc.coeffs()
    dt = rc.dtype
    prev, cur = top_pair
    out = [cur, prev]
    rest = blocks[1:] if rc.bootstrap == 'euler' else blocks
    s_new, s_cur = cur, prev
    for blk in reversed(list(rest)):
        s_prev = _unstep(alpha, beta, gamma, s_new, s_cur, blk(s_cur).to(dt))
        out.append(s_prev)
        s_new, s_cur = s_cur, s_prev
    return out[::-1]


class _RevStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s0, blocks, rc, *params):
        dt = rc.dtype
        with torch.no_grad():
            s0 = s0.to(dt)
            alpha, beta, gamma = rc.coeffs()
            if rc.bootstrap == 'euler':
                s1, rest = s0 + rc.h * blocks[0](s0).to(dt), blocks[1:]
            else:
                s1, rest = s0, blocks
            prev, cur = s0, s1
            for blk in rest:
                prev, cur = cur, _step(alpha, beta, gamma, prev, cur, blk(cur).to(dt))
        # The ONLY activations kept for backward: the top two states of the stack
        # (plus the bottom two when anchoring).
        if rc.anchor:
            ctx.save_for_backward(prev, cur, s0, s1)
        else:
            ctx.save_for_backward(prev, cur)
        ctx.n_states = len(rest) + 2
        ctx.blocks, ctx.rc = blocks, rc
        dev = s0.device.type
        ctx.ac = (dev, torch.is_autocast_enabled(dev), torch.get_autocast_dtype(dev))
        ctx.n_params = [len(list(b.parameters())) for b in blocks]
        return cur

    @staticmethod
    def backward(ctx, g_top):
        saved = ctx.saved_tensors
        prev, cur = saved[0], saved[1]
        anchors = {0: saved[2], 1: saved[3]} if len(saved) == 4 else {}
        blocks, rc = ctx.blocks, ctx.rc
        alpha, beta, gamma = rc.coeffs()
        dt = rc.dtype
        dev, ac_on, ac_dt = ctx.ac
        per_block = [None] * len(blocks)

        def vjp(blk, x, scale, g):
            """f(x) recomputed with a graph, then d/dx and d/dtheta of <scale * g, f(x)>."""
            x = x.detach().requires_grad_(True)
            ps = list(blk.parameters())
            with torch.enable_grad(), torch.autocast(dev, dtype=ac_dt, enabled=ac_on):
                fx = blk(x).to(dt)
            grads = torch.autograd.grad(fx, [x] + ps, grad_outputs=scale * g, allow_unused=True)
            return fx.detach(), grads[0], grads[1:]

        first = 1 if rc.bootstrap == 'euler' else 0
        s_new, s_cur = cur, prev
        i_cur = ctx.n_states - 2                       # index of s_cur in s[0..L]
        g_new, g_cur = g_top.to(dt), torch.zeros_like(prev)
        for k in range(len(blocks) - 1, first - 1, -1):
            # s_new = alpha*s_prev + beta*s_cur + gamma*f_k(s_cur)
            if i_cur in anchors:
                s_cur = anchors[i_cur]
            fx, gx, gp = vjp(blocks[k], s_cur, gamma, g_new)
            s_prev = _unstep(alpha, beta, gamma, s_new, s_cur, fx)
            i_cur -= 1                                 # s_prev is s[i_cur] now
            if i_cur in anchors:
                s_prev = anchors[i_cur]
            per_block[k] = gp
            g_cur_total = g_cur + gx + (beta * g_new if beta else 0)
            g_prev = alpha * g_new
            s_new, s_cur = s_cur, s_prev
            g_new, g_cur = g_cur_total, g_prev
            del fx, gx
        # now s_new = s[1], s_cur = s[0];  g_new = dL/ds1, g_cur = dL/ds0
        if 0 in anchors:
            s_cur = anchors[0]
        if rc.bootstrap == 'euler':
            _, gx, gp = vjp(blocks[0], s_cur, rc.h, g_new)
            per_block[0] = gp
            g0 = g_cur + g_new + gx
        else:
            g0 = g_cur + g_new
        flat = []
        for k, n in enumerate(ctx.n_params):
            flat.extend(per_block[k] if per_block[k] is not None else [None] * n)
        return (g0, None, None, *flat)


def rev_stack(blocks, s0, rc: RevCfg):
    params = [p for b in blocks for p in b.parameters()]
    return _RevStack.apply(s0, blocks, rc, *params)
