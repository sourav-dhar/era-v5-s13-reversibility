"""Memory and speed probes.  Each one runs real optimizer steps on random tokens, because peak
memory does not depend on what the tokens are.

    step_peak(cfg, B)          peak bytes of one steady-state training step, or None on OOM
    max_batch(cfg)             largest micro-batch that trains (doubling, then bisection)
    max_seq(cfg)               longest sequence that trains at batch 1
    throughput(cfg, batches)   tokens/s at each batch size
    depth_sweep(...)           peak memory against the number of layers
    timeline(cfg, B)           allocated bytes over one step, from the CUDA allocator's own trace
    drift(...)                 how far rebuilt states land from the true ones, and what that does to grads
"""
from __future__ import annotations

import math
import time

import torch

from .reversible import integrate, reconstruct
from .train import autocast, build, device, free, full_cfg, make_opt

GiB = 2 ** 30


def _one(c, B, T, n_steps, timed=False):
    dev = device()
    model = build(c, dev)
    opt = make_opt(model, c, dev)
    x = torch.randint(0, 50257, (B, T), device=dev)
    y = torch.randint(0, 50257, (B, T), device=dev)
    times = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for i in range(n_steps):
        t0 = time.perf_counter()
        with autocast(dev):
            loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated()
    return peak, times


def step_peak(cfg, B, T=None, n_steps=2, timed=False):
    c = full_cfg(cfg)
    T = T or c['seq']
    free()
    try:
        peak, times = _one(c, B, T, n_steps, timed)
        res = (peak, times)
    except torch.cuda.OutOfMemoryError:
        res = None
    free()
    return res


def max_batch(cfg, start=4, cap=8192):
    trials = {}

    def ok(B):
        r = step_peak(cfg, B)
        trials[B] = None if r is None else r[0]
        return r is not None

    lo, hi = 0, None
    B = start
    while B <= cap:
        if ok(B):
            lo, B = B, B * 2
        else:
            hi = B
            break
    if hi is None:
        return {'max_batch': lo, 'capped': True, 'peak': trials.get(lo), 'trials': trials}
    while hi - lo > max(1, lo // 64):          # bisect to within ~1.5%
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return {'max_batch': lo, 'capped': False, 'peak': trials.get(lo), 'trials': trials}


def max_seq(cfg, start=1024, cap=262144):
    c = full_cfg(cfg)
    trials, lo, hi, T = {}, 0, None, start
    while T <= cap:
        r = step_peak(c, 1, T=T, n_steps=1)
        trials[T] = None if r is None else r[0]
        if r is None:
            hi = T
            break
        lo, T = T, T * 2
    if hi is not None:
        while hi - lo > max(1024, lo // 16):      # long sequences are slow to probe: stop within ~6%
            mid = (lo + hi) // 2 // 1024 * 1024
            r = step_peak(c, 1, T=mid, n_steps=1)
            trials[mid] = None if r is None else r[0]
            lo, hi = (mid, hi) if r is not None else (lo, mid)
    return {'max_seq': lo, 'capped': hi is None, 'trials': trials}


def throughput(cfg, batches, n_steps=6, skip=2):
    out = []
    c = full_cfg(cfg)
    for B in batches:
        r = step_peak(c, B, n_steps=n_steps, timed=True)
        if r is None:
            out.append({'batch': B, 'oom': True})
            break
        peak, times = r
        t = sum(times[skip:]) / len(times[skip:])
        out.append({'batch': B, 'tok_per_s': B * c['seq'] / t, 'step_s': t, 'peak': peak})
    return out


def depth_sweep(cfg, depths, B):
    res = []
    for L in depths:
        r = step_peak(dict(cfg, n_layer=L), B)
        res.append({'n_layer': L, 'peak': None if r is None else r[0]})
    return res


# ---------------------------------------------------------------------------------------------
# Timeline: the allocator records every alloc/free; markers of odd sizes split the step in phases.
_MARK = 512 * 100_003


def _mark(k):
    t = torch.empty(_MARK + 512 * k, dtype=torch.uint8, device='cuda')
    del t


def timeline(cfg, B, points=900):
    c = full_cfg(cfg)
    dev = device()
    free()
    model = build(c, dev)
    opt = make_opt(model, c, dev)
    x = torch.randint(0, 50257, (B, c['seq']), device=dev)
    y = torch.randint(0, 50257, (B, c['seq']), device=dev)

    def step(marks):
        if marks: _mark(0)
        with autocast(dev):
            loss = model(x, y)
        if marks: _mark(1)
        loss.backward()
        if marks: _mark(2)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if marks: _mark(3)

    step(False)                       # optimizer state now exists: the next step is steady-state
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(max_entries=500_000)
    step(True)
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    trace = snap['device_traces'][torch.cuda.current_device()]
    cur, series, phase_at = base, [], {}
    for e in trace:
        a, s = e['action'], e['size']
        if a == 'alloc':
            if s >= _MARK and (s - _MARK) % 512 == 0 and (s - _MARK) // 512 < 4:
                phase_at[(s - _MARK) // 512] = len(series)
                continue
            cur += s
        elif a == 'free_completed':
            if s >= _MARK and (s - _MARK) % 512 == 0 and (s - _MARK) // 512 < 4:
                continue
            cur -= s
        else:
            continue
        series.append(cur)
    n = len(series)
    k = max(1, math.ceil(n / points))
    ds = [max(series[i:i + k]) for i in range(0, n, k)]       # bucket max keeps the peaks
    phases = {name: phase_at.get(i, 0) // k for i, name in enumerate(['forward', 'backward', 'optimizer', 'end'])}
    del model, opt, x, y
    free()
    return {'bytes': ds, 'phases': phases, 'base': base, 'peak': max(series) if series else base,
            'events': n}


# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def state_drift(model, idx, amp=True, state_dtype=None):
    """Relative error of every rebuilt state against the state the forward pass actually produced."""
    rc = model.cfg.rev
    if state_dtype:
        from dataclasses import replace
        rc = replace(rc, state_dtype=state_dtype)
    dev = idx.device
    ctx = autocast(dev) if amp else torch.autocast(dev.type, enabled=False)
    with ctx:
        s0 = model.wte(idx)
        top, states = integrate(model.blocks, s0, rc, keep_states=True)
        rebuilt = reconstruct(model.blocks, (states[-2], states[-1]), rc)
    err = []
    for a, b in zip(states, rebuilt):
        a, b = a.float(), b.float()
        err.append(((a - b).norm() / a.norm().clamp_min(1e-30)).item())
    norms = [s.float().norm(dim=-1).mean().item() for s in states]
    return {'rel_err': err, 'state_norm': norms}


def grad_fidelity(model, idx, tgt):
    """Cosine and relative error between the reversible-backward gradient and plain autograd's."""
    grads = {}
    for mode in ('rev_unrolled', 'rev'):
        model.cfg.mode = mode
        model.zero_grad(set_to_none=True)
        with autocast(idx.device):
            loss = model(idx, tgt)
        loss.backward()
        grads[mode] = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()}
    model.cfg.mode = 'rev'
    model.zero_grad(set_to_none=True)
    acc = {}                                   # layer -> [<g0,g1>, |g0|^2, |g1|^2, |g0-g1|^2]
    for n, g0 in grads['rev_unrolled'].items():
        g1 = grads['rev'][n]
        layer = int(n.split('.')[1]) if n.startswith('blocks.') else -1
        d = acc.setdefault(layer, [0.0, 0.0, 0.0, 0.0])
        d[0] += (g0 * g1).sum().item()
        d[1] += g0.pow(2).sum().item()
        d[2] += g1.pow(2).sum().item()
        d[3] += (g0 - g1).pow(2).sum().item()
    return [{'layer': L, 'cos': dot / math.sqrt(max(a * b, 1e-60)), 'rel_err': math.sqrt(dd / max(a, 1e-60))}
            for L, (dot, a, b, dd) in sorted(acc.items())]
