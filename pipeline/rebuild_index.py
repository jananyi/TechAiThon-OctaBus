import pickle
from pathlib import Path
from json import JSONDecoder

ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / 'global_kb.jsonl'
OUT = ROOT / 'pipeline' / 'output' / 'global_index.pkl'

keys = set()
dec = JSONDecoder()

if not GLOBAL.exists():
    print('global_kb.jsonl not found')
    raise SystemExit(1)

text = GLOBAL.read_text(encoding='utf-8')
for raw in text.splitlines():
    s = raw.strip()
    if not s:
        continue
    idx = 0
    L = len(s)
    while idx < L:
        try:
            obj, used = dec.raw_decode(s[idx:])
            topic = (obj.get('topic') or '').strip().lower()
            textv = (obj.get('text') or '').strip().lower()
            keys.add((topic, textv))
            idx += used
            while idx < L and s[idx].isspace():
                idx += 1
        except Exception:
            break

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('wb') as fh:
    pickle.dump(keys, fh)

print('Wrote', OUT, 'with', len(keys), 'keys')
