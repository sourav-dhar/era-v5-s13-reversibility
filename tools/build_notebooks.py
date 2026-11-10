"""Build notebooks/NN_*.ipynb from tools/cells/NN_*.md.

Cell markers (each on its own line):  %%md  |  %%code  |  %%setup   (the shared setup cell)
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MARK = re.compile(r'^%%(md|code|setup)\s*$')

SETUP = r'''# --- setup: works on Modal, on Colab (clones the repo) and locally -------------------------
import os, sys, json, math
from pathlib import Path
if 'google.colab' in sys.modules and not Path('/content/era-v5-s13-reversibility').exists():
    !git clone -q https://github.com/sourav-dhar/era-v5-s13-reversibility /content/era-v5-s13-reversibility
    !pip -q install tiktoken datasets
    os.environ.setdefault('S13_PROFILE', 'colab')
    os.chdir('/content/era-v5-s13-reversibility/notebooks')
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np, torch, matplotlib.pyplot as plt
from s13 import store, viz, probe
from s13.profiles import P
from s13.train import run, gpu_name, device, DEFAULTS
from s13.data import prepare, data_dir
viz.style()
GiB = 2**30
PROF, CFG = store.profile(), P()
B0 = dict(CFG['base'])
CUDA = torch.cuda.is_available()
print(f'profile = {PROF} | device = {gpu_name(device())} | torch {torch.__version__}')
print('data   :', prepare(data_dir(), **CFG['data']))'''


def build(src: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    nb.metadata['accelerator'] = 'GPU'
    kind, buf = None, []

    def flush():
        body = '\n'.join(buf).strip('\n')
        if kind == 'setup':
            nb.cells.append(nbf.v4.new_code_cell(SETUP))
        elif kind and body:
            nb.cells.append(nbf.v4.new_markdown_cell(body) if kind == 'md' else nbf.v4.new_code_cell(body))

    for line in src.read_text(encoding='utf-8').splitlines():
        m = MARK.match(line)
        if m:
            flush()
            kind, buf = m.group(1), []
        else:
            buf.append(line)
    flush()
    out = ROOT / 'notebooks' / (src.stem + '.ipynb')
    nbf.write(nb, out)
    return out


if __name__ == '__main__':
    srcs = sorted((ROOT / 'tools' / 'cells').glob('*.md'))
    for s in srcs:
        if len(sys.argv) == 1 or any(a in s.stem for a in sys.argv[1:]):
            print(build(s))
