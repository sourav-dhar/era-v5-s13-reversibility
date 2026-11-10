"""results/<profile>/*.json  ->  docs/data.js  (one global, S13 = {...}), trimmed for the web."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def thin(xs, n=240):
    if len(xs) <= n:
        return xs
    k = len(xs) / n
    return [xs[int(i * k)] for i in range(n)] + [xs[-1]]


def run_view(r):
    c = r['cfg']
    keep = ('name', 'mode', 'integrator', 'h', 'a', 'bootstrap', 'batch', 'lr', 'ce_chunk', 'tokens', 'anchor', 'seed')
    cur = r['curve']
    idx = thin(list(range(len(cur['tokens']))))
    return {
        'cfg': {k: c.get(k) for k in keep},
        'steps': r['steps'], 'final_val': r['final_val'], 'final_train': r['final_train'], 'diverged': r['diverged'],
        'tok_per_s': r['tok_per_s'], 'peak_gib': r['peak_mem_bytes'] / 2 ** 30, 'wall_s': r['wall_s'],
        'cost_usd': r['cost_usd'], 'gpu': r['gpu'],
        'curve': {'t': [cur['tokens'][i] for i in idx], 'loss': [cur['loss'][i] for i in idx],
                  'gnorm': [cur['gnorm'][i] for i in idx]},
        'evals': r['evals'], 'samples': r.get('samples', []),
    }


def main(profile='full'):
    src = ROOT / 'results' / profile
    out = {'profile': profile, 'runs': {}, 'probes': {}}
    for p in sorted(src.glob('*.json')):
        d = json.loads(p.read_text())
        k = p.stem
        if k.startswith('run_'):
            out['runs'][k[4:]] = run_view(d)
        elif k.startswith('timeline_'):
            d = dict(d)
            n = len(d['bytes'])
            step = max(1, n // 420)
            d['bytes'] = [max(d['bytes'][i:i + step]) / 2 ** 30 for i in range(0, n, step)]
            d['phases'] = {a: b // step for a, b in d['phases'].items()}
            d['peak'] /= 2 ** 30
            d['base'] /= 2 ** 30
            out['probes'][k] = d
        else:
            out['probes'][k] = d
    dst = ROOT / 'docs' / 'data.js'
    dst.parent.mkdir(exist_ok=True)
    dst.write_text('window.S13 = ' + json.dumps(out, separators=(',', ':'), default=float) + ';\n', encoding='utf-8')
    print(f'wrote {dst} ({dst.stat().st_size / 1024:.0f} KiB): {len(out["runs"])} runs, {len(out["probes"])} probes')


if __name__ == '__main__':
    main(*sys.argv[1:])
