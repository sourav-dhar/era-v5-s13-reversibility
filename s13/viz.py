"""One look for every figure: thin marks, recessive grid, colour fixed to the *entity*, never its rank.

Palette: the validated 8-slot categorical order (blue, orange, aqua, yellow, magenta, green, violet,
red).  Each memory mode / integrator owns one slot for the whole repo and the companion site.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e7e6e2', '#fcfcfb'

COLOR = {
    # memory modes
    'standard': SLOTS[0], 'ckpt': SLOTS[1], 'rev': SLOTS[2], 'rev_unrolled': SLOTS[3],
    # integrators (reversible family)
    'midpoint': SLOTS[2], 'blend': SLOTS[4], 'leapfrog': SLOTS[6],
}
LABEL = {'standard': 'standard (stores activations)', 'ckpt': 'activation checkpointing',
         'rev': 'reversible (custom backward)', 'rev_unrolled': 'reversible rule, plain autograd'}
DASH = {'euler': '-', 'none': (0, (4, 2))}


def style():
    mpl.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight',
        'font.size': 10, 'axes.titlesize': 11.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.labelcolor': INK2, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 0, 'ytick.major.size': 0,
        'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round',
        'legend.frameon': False, 'legend.fontsize': 9, 'text.color': INK,
        'axes.prop_cycle': mpl.cycler(color=SLOTS),
    })


def save(fig, name, root=None):
    root = Path(root or os.environ.get('S13_FIGURES') or Path(__file__).resolve().parents[1] / 'figures')
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f'{name}.png')
    return root / f'{name}.png'


def rev_color(cfg):
    if cfg.get('mode') == 'rev' or cfg.get('mode') == 'rev_unrolled':
        return COLOR.get(cfg.get('integrator', 'midpoint'), SLOTS[2])
    return COLOR.get(cfg.get('mode'), INK2)


def gib(b):
    return None if b is None else b / 2 ** 30


def ema(xs, a=0.9):
    out, m = [], None
    for x in xs:
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out
