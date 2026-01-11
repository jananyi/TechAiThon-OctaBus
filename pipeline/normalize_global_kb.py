"""Normalize existing global_kb.jsonl to ensure each JSON object has ordered keys.

Writes a backup to `global_kb.jsonl.bak` and rewrites `global_kb.jsonl`.
"""
import json
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / 'global_kb.jsonl'
BACKUP = ROOT / 'global_kb.jsonl.bak'

if not GLOBAL.exists():
    print('global_kb.jsonl not found')
    raise SystemExit(1)

lines = GLOBAL.read_text(encoding='utf-8').splitlines()
objs = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
        objs.append(o)
    except Exception:
        continue

BACKUP.write_text('\n'.join(lines), encoding='utf-8')

with GLOBAL.open('w', encoding='utf-8') as fh:
    for o in objs:
        out = OrderedDict()
        out['id'] = int(o.get('id')) if o.get('id') is not None else None
        out['topic'] = o.get('topic','')
        out['text'] = o.get('text','')
        out['treatment date'] = o.get('treatment date','')
        out['treatment country'] = o.get('treatment country','')
        out['treatment given'] = o.get('treatment given','')
        fh.write(json.dumps(out, ensure_ascii=False) + '\n')

print('Normalized global_kb.jsonl and wrote backup to', BACKUP)
