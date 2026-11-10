"""Everything that needs a GPU runs here, on a Modal A100-40GB.

    modal run tools/modal_app.py --task prep                       # FineWeb-Edu -> 52M+2M GPT-2 tokens (CPU)
    modal run tools/modal_app.py --task smoke                      # every code path on the GPU in ~3 min
    modal run tools/modal_app.py --task nb --names 01,02           # execute notebooks (in parallel)
    modal volume get era-v5-s13 results/full results/ --force      # pull the results back

Notebooks are executed in a container that holds this repo, with the data, results and
checkpoints on a shared volume. The executed notebook (with every output) lands in
/vol/notebooks/ and is pulled back into notebooks/.
"""
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.8.0', 'numpy', 'tiktoken', 'datasets', 'matplotlib', 'pandas',
                      'nbformat', 'nbconvert', 'ipykernel', 'matplotlib-inline')
         .add_local_dir(ROOT, '/root/repo', ignore=['.venv', '.git', 'data', 'checkpoints', 'results',
                                                     'docs', 'figures', '**/__pycache__']))

app = modal.App('era-v5-s13-reversibility', image=image)
vol = modal.Volume.from_name('era-v5-s13', create_if_missing=True)
ENV = {'S13_DATA': '/vol/data/fwe', 'S13_RESULTS': '/vol/results', 'S13_CKPT': '/vol/ckpt', 'S13_FIGURES': '/vol/figures', 'S13_MEM_CAP_GIB': '39.5',
       'PYTHONPATH': '/root/repo', 'HF_HUB_DISABLE_PROGRESS_BARS': '1'}


@app.function(cpu=8, memory=16384, timeout=3 * 3600, volumes={'/vol': vol})
def prep():
    os.environ.update(ENV)
    sys.path.insert(0, '/root/repo')
    from s13.data import prepare
    meta = prepare('/vol/data/fwe')
    vol.commit()
    return meta


def wrong_gpu():
    """Modal sometimes serves an 'A100-40GB' request on an 80GB card. Memory limits and tokens/s would then
    not be comparable, so every GPU job refuses such a card and asks the caller to retry."""
    import torch
    big = torch.cuda.get_device_properties(0).total_memory > 45 * 2**30
    if big:
        try:
            modal.experimental.stop_fetching_inputs()
        except Exception:
            pass
    return big


@app.function(gpu='A100-40GB', timeout=1800, volumes={'/vol': vol})
def smoke():
    os.environ.update(ENV)
    os.environ['S13_PROFILE'] = 'smoke'
    sys.path.insert(0, '/root/repo')
    import torch
    from s13.train import run
    from s13 import probe
    out = {}
    base = dict(tokens=3_000_000, batch=16, eval_every=1_000_000, eval_tokens=65536, final_eval_tokens=131072,
                samples=False)
    for name, extra in [('std', {}), ('ckpt', {'mode': 'ckpt'}), ('rev_mid', {'mode': 'rev'}),
                        ('rev_blend', {'mode': 'rev', 'integrator': 'blend', 'h': 0.25, 'a': 0.5}),
                        ('rev_leap', {'mode': 'rev', 'integrator': 'leapfrog', 'h': 1.0}),
                        ('rev_chunk', {'mode': 'rev', 'ce_chunk': 4096})]:
        r = run(dict(base, name=name, **extra))
        out[name] = {k: r[k] for k in ('final_val', 'tok_per_s', 'peak_mem_bytes', 'wall_s')}
    out['peak_B32'] = {m: probe.step_peak({'mode': m}, 32)[0] / 2**30 for m in ('standard', 'ckpt', 'rev_unrolled', 'rev')}
    tl = probe.timeline({'mode': 'rev'}, 8)
    out['timeline'] = {'events': tl['events'], 'phases': tl['phases'], 'peak': tl['peak'] / 2**30, 'n': len(tl['bytes'])}
    m = __import__('s13.train', fromlist=['build']).build(dict(__import__('s13.train', fromlist=['DEFAULTS']).DEFAULTS, mode='rev'), torch.device('cuda'))
    idx = torch.randint(0, 50257, (4, 1024), device='cuda')
    out['drift'] = probe.state_drift(m, idx)['rel_err']
    out['gradfid'] = probe.grad_fidelity(m, idx, idx)
    return out


RETRY = modal.Retries(max_retries=6, initial_delay=2.0, backoff_coefficient=1.0)


@app.function(gpu='A100-40GB', timeout=6 * 3600, volumes={'/vol': vol}, retries=RETRY)
def notebook(name: str, profile: str = 'full'):
    import subprocess
    if wrong_gpu():
        raise RuntimeError('served an 80GB A100; Modal retries on another container')
    os.environ.update(ENV)
    os.environ['S13_PROFILE'] = profile
    os.makedirs('/vol/notebooks', exist_ok=True)
    src = next(Path('/root/repo/notebooks').glob(f'{name}*.ipynb'))
    out = Path('/vol/notebooks') / src.name
    cmd = ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', str(src), '--output', str(out),
           '--ExecutePreprocessor.timeout=-1', '--ExecutePreprocessor.kernel_name=python3']
    p = subprocess.run(cmd, cwd='/root/repo/notebooks', capture_output=True, text=True)
    vol.commit()
    return {'name': src.name, 'returncode': p.returncode, 'stderr': p.stderr[-4000:]}


@app.function(gpu='A100-40GB', timeout=2 * 3600, volumes={'/vol': vol}, retries=RETRY)
def train_cached(key: str, cfg: dict, profile: str = 'full'):
    """Train one config and store it under the same key a notebook's store.cached() would use."""
    if wrong_gpu():
        raise RuntimeError('served an 80GB A100; Modal retries on another container')
    os.environ.update(ENV)
    os.environ['S13_PROFILE'] = profile
    sys.path.insert(0, '/root/repo')
    from s13 import store
    from s13.train import run
    r = store.cached(key, lambda: run(cfg))
    vol.commit()
    return key, r['final_val'], r['tok_per_s']


@app.function(gpu='A100-40GB', timeout=3 * 3600, volumes={'/vol': vol}, retries=RETRY)
def maxseq_cached(mode: str, profile: str = 'full'):
    """One longest-sequence probe, stored under the key notebook 04 reads."""
    if wrong_gpu():
        raise RuntimeError('served an 80GB A100; Modal retries on another container')
    os.environ.update(ENV)
    os.environ['S13_PROFILE'] = profile
    sys.path.insert(0, '/root/repo')
    from s13 import store, probe
    from s13.profiles import PROFILES
    P = PROFILES[profile]
    rev = {}
    if mode == 'rev':
        w = store.load('winner')
        rev = {k: w[k] for k in ('integrator', 'h', 'a', 'bootstrap')}
    r = store.cached(f'maxseq_{mode}', lambda: probe.max_seq(dict(P['base'], mode=mode, ce_chunk=4096, **rev), cap=P['seq_cap']))
    vol.commit()
    return mode, r['max_seq'], r['capped']


@app.local_entrypoint()
def main(task: str = 'smoke', names: str = '', profile: str = 'full'):
    import json
    if task == 'prep':
        print(json.dumps(prep.remote(), indent=1))
    elif task == 'smoke':
        print(json.dumps(smoke.remote(), indent=1, default=float))
    elif task in ('ablation', 'anchored'):
        from s13.profiles import PROFILES
        P = PROFILES[profile]
        jobs = [(f'run_rev-{n}', dict(P['base'], name=f'rev-{n}', mode='rev', integrator=i, h=h, a=a, bootstrap=b,
                                      anchor=task == 'anchored', batch=32, tokens=P['tokens']), profile)
                for n, i, h, a, b in P[task]]
        for attempt in range(12):
            retry = []
            for k, v, t in train_cached.starmap(jobs, order_outputs=False):
                if v is None:
                    retry.append(next(j for j in jobs if j[0] == k)); print(f'{k}: 80GB card, retrying')
                else:
                    print(f'{k}: val {v:.4f}  {t:,.0f} tok/s')
            jobs = retry
            if not jobs:
                break
    elif task == 'seeds':
        from s13.profiles import PROFILES
        P = PROFILES[profile]
        base = dict(P['base'], batch=32, tokens=P['tokens'], seed=1)
        jobs = [('run_baseline-s1', dict(base, name='baseline-s1', mode='standard'), profile),
                ('run_rev-anc-leapfrog-euler-s1', dict(base, name='rev-anc-leapfrog-euler-s1', mode='rev', integrator='leapfrog',
                                                       h=1.0, bootstrap='euler', anchor=True), profile),
                ('run_rev-midpoint-nokick-s1', dict(base, name='rev-midpoint-nokick-s1', mode='rev', integrator='midpoint',
                                                    h=0.5, bootstrap='none'), profile)]
        for attempt in range(12):
            retry = []
            for k, v, t in train_cached.starmap(jobs, order_outputs=False):
                if v is None:
                    retry.append(next(j for j in jobs if j[0] == k)); print(f'{k}: 80GB card, retrying')
                else:
                    print(f'{k}: val {v:.4f}  {t:,.0f} tok/s')
            jobs = retry
            if not jobs:
                break
    elif task == 'nb':
        todo = [n.strip() for n in names.split(',') if n.strip()]
        for attempt in range(12):
            if not todo:
                break
            retry = []
            for r in notebook.starmap([(n, profile) for n in todo], order_outputs=False):
                if r['returncode'] == 99:
                    print(f"{r['name']}: got an 80GB card, retrying"); retry.append(r['name'][:2] if r['name'][:2].isdigit() else r['name'])
                    continue
                print(f"{r['name']}: rc={r['returncode']}")
                if r['returncode']:
                    print(r['stderr'])
            todo = retry
