"""Fill the README's data-driven tables from results/full/*.json, so no number in them is typed by hand.

    python tools/fill_readme.py          # replaces each ⟪KEY⟫ marker that has data; reports the rest
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / 'full'
GiB = 2 ** 30


def load(k):
    p = RES / f'{k}.json'
    return json.loads(p.read_text()) if p.exists() else None


def run(k):
    return load(f'run_{k}')


def label(r):
    c = r['cfg']
    s = 'standard' if c['mode'] == 'standard' else f"{c['integrator']} h={c['h']:g}"
    if c['mode'] == 'rev' and c['integrator'] == 'blend':
        s = f"blend a={c['a']:g}, h={c['h']:g}"
    if c['mode'] == 'rev':
        s = f"reversible {s}, {'Euler' if c['bootstrap'] == 'euler' else 'no-kick'} start"
    if c.get('ce_chunk'):
        s += ' + chunked head'
    return s


def fill():
    md = (ROOT / 'README.md').read_text(encoding='utf-8')
    base = run('baseline')
    F = {}

    mb = [k for k in ('std-max', 'std-max-sqrtlr', 'rev-max', 'rev-max-sqrtlr', 'rev-chunk-max', 'rev-chunk-max-sqrtlr') if run(k)]
    if 'rev-chunk-max' in mb:
        rows = ['| run | what | micro-batch | tokens / step | steps | LR | val loss | Δ vs baseline | tokens/s | peak |',
                '|---|---|--:|--:|--:|--:|--:|--:|--:|--:|']
        for k in ['baseline', 'rev-midpoint-nokick'] + mb:
            r = run(k)
            rows.append(f"| `{k}` | {label(r)} | {r['cfg']['batch']} | {r['tokens_per_step']:,} | {r['steps']:,} | {r['cfg']['lr']:.2e} | "
                        f"{r['final_val']:.4f} | {r['final_val'] - base['final_val']:+.4f} | {r['tok_per_s']:,.0f} | {r['peak_mem_bytes'] / GiB:.2f} GiB |")
        F['MB_TABLE'] = '\n'.join(rows)
        best = min((run(k) for k in ('rev-chunk-max', 'rev-chunk-max-sqrtlr')), key=lambda r: r['final_val'])
        F['MB_WHAT'] = f"midpoint, no-kick start + chunked head{', √-scaled LR' if 'sqrt' in best['cfg']['name'] else ''}"
        F['MB_BATCH'] = f"{best['cfg']['batch']:,} × 1,024"
        F['MB_STEPS'] = f"{best['steps']:,}"
        F['MB_VAL'] = f"{best['final_val']:.4f}"
        F['MB_TRAIN'] = f"{best['final_train']:.4f}"
        F['MB_TPS'] = f"{best['tok_per_s']:,.0f}"
        F['MB_PEAK'] = f"{best['peak_mem_bytes'] / GiB:.2f} GiB"
        F['MB_COST'] = f"{best['wall_s']:.0f} s · ${best['cost_usd']:.2f}"
        F['MB_BEST'] = best

    seq = {m: load(f'maxseq_{m}') for m in ('standard', 'ckpt', 'rev')}
    if all(seq.values()):
        f = lambda d: f"**{d['max_seq']:,} tokens**" + (' (the probe cap; it never ran out of memory)' if d['capped'] else '')
        F['SEQ_CKPT'], F['SEQ_REV'] = f(seq['ckpt']), f(seq['rev'])

    anc = {n: load(f'drift_anchor_{n}') for n in ('midpoint-nokick', 'midpoint-euler', 'blend-euler', 'blend-nokick', 'leapfrog-euler', 'leapfrog-nokick', 'blend-libdefault')}
    if all(anc.values()):
        rows = ['| trained model | worst-layer cosine, rebuilt | **anchored** | layer-0 cosine, rebuilt | **anchored** |', '|---|--:|--:|--:|--:|']
        for n, a in anc.items():
            d = load(f'drift_{n}')
            g0 = {r['layer']: r['cos'] for r in d['grad']}
            g1 = {r['layer']: r['cos'] for r in a['grad_anchored']}
            rows.append(f"| {n} | {min(g0.values()):.4f} | **{min(g1.values()):.4f}** | {g0[0]:.4f} | **{g1[0]:.4f}** |")
        F['ANCHOR_FID'] = '\n'.join(rows)

    runs = sorted((json.loads(p.read_text()) for p in RES.glob('run_*.json')), key=lambda r: r['wall_s'])
    if runs:
        rows = ['| run | GPU | wall time | cost |', '|---|---|--:|--:|']
        for r in runs:
            rows.append(f"| `{r['cfg']['name']}` | {r['gpu'].replace('NVIDIA ', '')} | {r['wall_s']:.0f} s | ${r['cost_usd']:.3f} |")
        tot_s, tot = sum(r['wall_s'] for r in runs), sum(r['cost_usd'] for r in runs)
        rows.append(f"| **all {len(runs)} training runs** | | **{tot_s / 60:.0f} min** | **${tot:.2f}** |")
        F['COST_TABLE'] = '\n'.join(rows)

    for k, v in F.items():
        if isinstance(v, str):
            md = md.replace(f'⟪{k}⟫', v)
    (ROOT / 'README.md').write_text(md, encoding='utf-8')
    left = sorted(set(part.split('⟫')[0] for part in md.split('⟪')[1:]))
    print('filled:', [k for k, v in F.items() if isinstance(v, str)])
    print('still open:', left)
    return F


if __name__ == '__main__':
    fill()
