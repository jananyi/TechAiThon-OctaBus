from pathlib import Path
from llama_local import rephrase_with_model

ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / 'global_kb.jsonl'

def sample_context():
    # find a short entry to use as context
    if not GLOBAL.exists():
        return 'No dataset available.'
    with GLOBAL.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                import json
                o = json.loads(line)
                return o.get('text','')[:400]
            except Exception:
                continue
    return ''

if __name__ == '__main__':
    ctx = sample_context()
    q = 'What is the initial treatment approach for chest pain?'
    print('Using context sample:', ctx[:120])
    out = rephrase_with_model(q, ctx)
    print('Model output:\n', out)
