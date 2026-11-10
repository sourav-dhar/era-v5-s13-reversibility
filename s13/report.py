"""Tables and figures shared by the notebooks."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from . import viz
from .viz import GRID, INK2, MUTED, gib


def label(r):
    c = r['cfg']
    if c['mode'] in ('rev', 'rev_unrolled'):
        if c['integrator'] == 'blend':
            core = f"blend a={c['a']:g}, h={c['h']:g}"
        else:
            core = f"{c['integrator']} h={c['h']:g}"
        s = f"{core}, {'Euler' if c['bootstrap'] == 'euler' else 'no-kick'} start"
    else:
        s = viz.LABEL.get(c['mode'], c['mode'])
    if c.get('ce_chunk'):
        s += ' + chunked head'
    return s


def row(r):
    c = r['cfg']
    return {
        'run': c['name'], 'what': label(r), 'batch (seq)': c['batch'], 'tokens/step': r['tokens_per_step'],
        'steps': r['steps'], 'final val loss': round(r['final_val'], 4), 'final train loss': round(r['final_train'], 4),
        'tokens/s': int(r['tok_per_s']), 'peak mem (GiB)': round(gib(r['peak_mem_bytes']), 2),
        'wall (s)': int(r['wall_s']), 'cost ($)': round(r['cost_usd'], 3),
    }


def table(runs):
    import pandas as pd
    return pd.DataFrame([row(r) for r in runs]).set_index('run')


def curves(runs, title='', ax=None, colors=None, styles=None, ymax=None, show_train=True):
    ax = ax or plt.subplots(figsize=(8.6, 4.4))[1]
    for i, r in enumerate(runs):
        col = (colors or {}).get(r['cfg']['name'], viz.rev_color(r['cfg']))
        ls = (styles or {}).get(r['cfg']['name'], viz.DASH.get(r['cfg'].get('bootstrap'), '-')
                                 if r['cfg']['mode'].startswith('rev') else '-')
        t = np.array(r['curve']['tokens']) / 1e6
        if show_train:
            ax.plot(t, viz.ema(r['curve']['loss'], 0.8), color=col, lw=1.2, alpha=0.35, ls=ls)
        et = np.array(r['evals']['tokens']) / 1e6
        ax.plot(et, r['evals']['val'], color=col, lw=2, ls=ls, marker='o', ms=4, label=f"{label(r)}  →  {r['final_val']:.3f}")
    ax.set_xlabel('tokens seen (millions)')
    ax.set_ylabel('loss  (faint: train, markers: validation)')
    if ymax:
        ax.set_ylim(top=ymax)
    ax.set_title(title)
    ax.legend(loc='upper right')
    return ax


def hbars(ax, names, values, colors, fmt='{:.1f}', xlabel='', annotate=True):
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, color=colors, height=0.62, edgecolor='white', linewidth=2)
    ax.set_yticks(y, names)
    ax.grid(axis='y', visible=False)
    if annotate:
        for yi, v in zip(y, values):
            ax.text(v, yi, '  ' + fmt.format(v), va='center', ha='left', color=INK2, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.margins(x=0.18)
    return ax


def char_roots(alpha, beta):
    """Roots of the linear (f = 0) recurrence  z^2 = beta z + alpha.  Forward growth per layer is max|z|;
    running it backward the roots are 1/z, so backward growth per layer is max|1/z|."""
    r = np.roots([1.0, -beta, -alpha])
    return r, max(abs(r)), max(abs(1 / r))
