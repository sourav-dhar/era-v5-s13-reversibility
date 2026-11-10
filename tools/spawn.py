"""Fire-and-forget: spawn jobs on the deployed app so they never depend on this machine staying up.

    modal deploy tools/modal_app.py
    python tools/spawn.py nb:01 nb:02 nb:03 nb:04 maxseq:ckpt maxseq:rev
"""
import sys

import modal

APP = 'era-v5-s13-reversibility'
for job in sys.argv[1:]:
    kind, arg = job.split(':', 1)
    if kind == 'nb':
        c = modal.Function.from_name(APP, 'notebook').spawn(arg, 'full')
    elif kind == 'maxseq':
        c = modal.Function.from_name(APP, 'maxseq_cached').spawn(arg, 'full')
    else:
        raise SystemExit(f'unknown job {job}')
    print(job, c.object_id)
