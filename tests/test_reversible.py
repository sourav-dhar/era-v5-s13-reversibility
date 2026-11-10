"""The reversible backward must be *the same gradient* as ordinary autograd, not an approximation of it.

In float64 the reconstruction error is ~1e-15, so the custom backward has to agree with autograd on
the unrolled integrator to ~1e-10 for every integrator and bootstrap.  It also has to agree with the
standard model when the stack is configured to *be* the standard model.
"""
import itertools

import pytest
import torch

from s13.model import GPT, ModelCfg
from s13.reversible import RevCfg, integrate, reconstruct

TINY = dict(vocab=97, n_layer=5, width=32, n_head=4)
VARIANTS = [RevCfg('midpoint', h=0.5, bootstrap=b, state_dtype='float64') for b in ('euler', 'none')] + \
           [RevCfg('blend', h=0.25, a=0.5, bootstrap=b, state_dtype='float64') for b in ('euler', 'none')] + \
           [RevCfg('leapfrog', h=0.7, bootstrap='euler', state_dtype='float64')] + \
           [RevCfg(i, h=0.4, a=0.5, bootstrap=b, state_dtype='float64', anchor=True)
            for i in ('midpoint', 'blend', 'leapfrog') for b in ('euler', 'none')]


def _grads(mode, rc, ce_chunk=0, seed=0):
    torch.manual_seed(seed)
    m = GPT(ModelCfg(**TINY, mode=mode, rev=rc, ce_chunk=ce_chunk)).double()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 97, (3, 16), generator=g)
    y = torch.randint(0, 97, (3, 16), generator=g)
    loss = m(x, y)
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in m.named_parameters()}


@pytest.mark.parametrize('rc', VARIANTS, ids=lambda r: r.label())
def test_rev_backward_equals_autograd(rc):
    l0, g0 = _grads('rev_unrolled', rc)
    l1, g1 = _grads('rev', rc)
    assert abs(l0 - l1) < 1e-12
    for n in g0:
        err = (g0[n] - g1[n]).abs().max().item() / (g0[n].abs().max().item() + 1e-30)
        assert err < 1e-9, (n, err)


def test_reconstruction_is_exact_in_fp64():
    torch.manual_seed(0)
    rc = VARIANTS[0]
    m = GPT(ModelCfg(**TINY, mode='rev', rev=rc)).double()
    s0 = m.wte(torch.randint(0, 97, (2, 16)))
    with torch.no_grad():
        top, states = integrate(m.blocks, s0, rc, keep_states=True)
        rebuilt = reconstruct(m.blocks, (states[-2], states[-1]), rc)
    assert len(rebuilt) == len(states)
    for a, b in zip(states, rebuilt):
        assert (a - b).abs().max() < 1e-12


def test_checkpoint_and_chunked_loss_match_standard():
    l0, g0 = _grads('standard', RevCfg(state_dtype='float64'))
    for mode, ch in [('ckpt', 0), ('standard', 7)]:
        l1, g1 = _grads(mode, RevCfg(state_dtype='float64'), ce_chunk=ch)
        assert abs(l0 - l1) < 1e-12
        for n in g0:
            assert torch.allclose(g0[n], g1[n], rtol=1e-9, atol=1e-12), (mode, n)


def test_param_count_is_20m_class():
    m = GPT(ModelCfg())
    assert 20e6 < m.n_params() < 22e6
