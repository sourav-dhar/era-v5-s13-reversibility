"""FineWeb-Edu, GPT-2 BPE, flat uint16 token files.

    prepare('/data/fwe')    ->  /data/fwe/{train,val}.bin  +  meta.json

The first documents of the stream become the validation set and are never trained on.

``Windows`` turns a token file into non-overlapping windows of T+1 tokens and visits them in one
fixed, seeded order.  A run with batch B takes windows [o, o+B) where o is the number of sequences
already seen, so **every run, whatever its batch size, sees exactly the same tokens in exactly the
same order**.  Only the grouping into optimizer steps differs.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def prepare(out_dir, train_tokens=52_000_000, val_tokens=2_000_000,
            repo='HuggingFaceFW/fineweb-edu', subset='sample-10BT', batch_docs=512):
    import tiktoken
    from datasets import load_dataset
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'meta.json').exists():
        return json.loads((out / 'meta.json').read_text())
    enc = tiktoken.get_encoding('gpt2')
    eot = enc.eot_token
    ds = load_dataset(repo, name=subset, split='train', streaming=True)
    t0 = time.time()
    targets = [('val', val_tokens), ('train', train_tokens)]
    meta, it, buf_docs = {'repo': repo, 'subset': subset, 'tokenizer': 'gpt2', 'eot': eot}, iter(ds), []
    for split, want in targets:
        arr, n, docs = np.empty(want, dtype=np.uint16), 0, 0
        while n < want:
            if not buf_docs:
                buf_docs = [next(it)['text'] for _ in range(batch_docs)]
            toks = enc.encode_ordinary_batch(buf_docs)
            buf_docs = []
            for t in toks:
                t = t + [eot]
                k = min(len(t), want - n)
                arr[n:n + k] = t[:k]
                n += k
                docs += 1
                if n >= want:
                    break
        arr.tofile(out / f'{split}.bin')
        meta[split] = {'tokens': int(n), 'docs': docs}
        print(f'[data] {split}: {n:,} tokens from {docs:,} documents ({time.time() - t0:.0f}s)')
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    return meta


class Windows:
    def __init__(self, path, T, seed=1234):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.T = T
        self.n = (len(self.data) - 1) // T
        self.order = np.random.default_rng(seed).permutation(self.n)

    def get(self, offset, B, device):
        idx = self.order[(offset + np.arange(B)) % self.n]
        w = np.stack([self.data[i * self.T: i * self.T + self.T + 1] for i in idx]).astype(np.int64)
        w = torch.from_numpy(w)
        if device.type == 'cuda':
            w = w.pin_memory().to(device, non_blocking=True)
        else:
            w = w.to(device)
        return w[:, :-1], w[:, 1:]


def data_dir():
    from .store import profile
    return Path(os.environ.get('S13_DATA', Path(__file__).resolve().parents[1] / 'data' / f'fwe-{profile()}'))
