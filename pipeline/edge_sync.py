"""Edge sync worker: watches for global updates and syncs global.pt into an edge store,
 then rebuilds a local index for fast deduplication on the edge side.
"""
import time
from pathlib import Path
import io
import torch
from encrypt_utils import load_key, decrypt_bytes
import pickle

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE_ROOT / 'pipeline' / 'output'
EDGE_STORE = WORKSPACE_ROOT / 'pipeline' / 'edge_store'
EDGE_STORE.mkdir(parents=True, exist_ok=True)
INDEX_PATH = EDGE_STORE / 'global_index.pkl'


def sync_once():
    flag = OUTPUT_DIR / 'global_sync.flag'
    if not flag.exists():
        return False
    # read global.pt (encrypted) and decrypt using global.key if present
    global_pt = OUTPUT_DIR / 'global.pt'
    global_key = OUTPUT_DIR / 'global.key'
    if not global_pt.exists() or not global_key.exists():
        return False
    try:
        cipher = global_pt.read_bytes()
        key = load_key(global_key)
        plain = decrypt_bytes(cipher, key)
        # save encrypted copy to edge store
        (EDGE_STORE / 'global.pt').write_bytes(cipher)
        # load torch object
        all_global = torch.load(io.BytesIO(plain))
        # build index
        keys = set()
        for e in all_global:
            topic = (e.get('topic') or '').strip().lower()
            text = (e.get('text') or '').strip().lower()
            keys.add((topic, text))
        # persist index
        with INDEX_PATH.open('wb') as fh:
            pickle.dump(keys, fh)
        # touch local sync marker
        (EDGE_STORE / 'sync.flag').write_text(str(time.time()))
        return True
    except Exception:
        return False


def run_poll(interval: float = 2.0):
    last = 0
    while True:
        try:
            flag = OUTPUT_DIR / 'global_sync.flag'
            if flag.exists():
                ts = flag.stat().st_mtime
                if ts != last:
                    if sync_once():
                        last = ts
        except Exception:
            pass
        time.sleep(interval)


if __name__ == '__main__':
    run_poll()
