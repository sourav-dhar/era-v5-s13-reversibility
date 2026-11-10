"""Three sizes of the same study.

    full   the reported runs: 20M GPT, 50M tokens per run, A100-40GB (Modal)
    colab  the same 50M-token runs on a free Colab T4, with the variant study cut to the two
           integrators the brief names and smaller probes
    quick  a tiny model on 1M tokens; every cell runs on a CPU in a few minutes (smoke test)
"""
from .store import profile

_FULL = dict(
    base=dict(),
    data=dict(train_tokens=52_000_000, val_tokens=2_000_000),
    tokens=50_000_000,
    variants=[
        # name                integrator   h     a     bootstrap
        ('midpoint-euler',   'midpoint', 0.5,  0.5,  'euler'),
        ('midpoint-nokick',  'midpoint', 0.5,  0.5,  'none'),
        ('blend-euler',      'blend',    0.25, 0.5,  'euler'),      # Lightning LM production settings
        ('blend-nokick',     'blend',    0.25, 0.5,  'none'),
        ('blend-libdefault', 'blend',    0.05, 0.95, 'none'),       # the integrator library's defaults
        ('leapfrog-euler',   'leapfrog', 1.0,  0.5,  'euler'),
        ('leapfrog-nokick',  'leapfrog', 1.0,  0.5,  'none'),
    ],
    # step-size ablation: does blend lose because of its averaging or because its effective step is 4x smaller?
    # the effective per-layer weight on f is gamma: midpoint 2h, blend h.
    ablation=[
        ('abl-blend-h1',       'blend',    1.0,  0.5,  'none'),     # gamma = 1, same as midpoint h=0.5
        ('abl-midpoint-h0.25', 'midpoint', 0.25, 0.5,  'none'),     # gamma = 0.5
        ('abl-midpoint-h1',    'midpoint', 1.0,  0.5,  'none'),     # gamma = 2
    ],
    # anchoring: keep s[0], s[1] as well, so the tiny bottom states are never rebuilt by subtraction
    anchored=[
        ('anc-midpoint-nokick', 'midpoint', 0.5,  0.5, 'none'),
        ('anc-blend-euler',     'blend',    0.25, 0.5, 'euler'),
        ('anc-leapfrog-euler',  'leapfrog', 1.0,  0.5, 'euler'),
    ],
    depths=[2, 4, 8, 12, 16, 24, 32, 48, 64],
    depth_batch=8,
    seq_cap=1_048_576,
    tput_batches=[4, 8, 16, 32, 64, 128, 256, 512],
    batch_cap=8192,
)

_COLAB = dict(_FULL, variants=[v for v in _FULL['variants'] if v[0] in ('midpoint-euler', 'blend-euler')],
              depths=[2, 4, 8, 12, 16, 24, 32], seq_cap=65_536, tput_batches=[4, 8, 16, 32, 64], batch_cap=2048)

_QUICK = dict(_FULL, base=dict(n_layer=4, width=128, n_head=4, seq=128, eval_every=250_000, eval_tokens=16_384,
                               final_eval_tokens=32_768, eval_batch=8),
              data=dict(train_tokens=1_200_000, val_tokens=200_000), tokens=1_000_000, variants=_FULL['variants'][:3], depths=[2, 4, 8], depth_batch=2,
              seq_cap=2048, tput_batches=[2, 4], batch_cap=64)

PROFILES = {'full': _FULL, 'colab': _COLAB, 'quick': _QUICK, 'smoke': _QUICK}


def P():
    return PROFILES[profile()]
