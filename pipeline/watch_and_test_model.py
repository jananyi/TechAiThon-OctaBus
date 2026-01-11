import time
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
REPO_NAME = 'TinyLlama-1.1B-Chat-v1.0'
MODEL_DIR = MODELS_DIR / REPO_NAME
CFG = ROOT / 'pipeline' / 'llama_config.json'

print('Watcher started: waiting for', MODEL_DIR)
timeout = 60 * 60  # 1 hour
poll = 10
elapsed = 0
while elapsed < timeout:
    if MODEL_DIR.exists():
        # check for a weights file
        has_weights = any(p.suffix in ('.safetensors', '.bin') for p in MODEL_DIR.rglob('*') if p.is_file())
        if has_weights:
            print('Model files detected in', MODEL_DIR)
            # update config to point to local model dir
            try:
                cfg = json.loads(CFG.read_text(encoding='utf-8')) if CFG.exists() else {}
                cfg['model_name'] = str(MODEL_DIR)
                CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
                print('Updated', CFG)
            except Exception as e:
                print('Failed updating config:', e)

            # run test script
            print('Running test_llama.py')
            try:
                subprocess.run(['python', str(ROOT / 'pipeline' / 'test_llama.py')], check=False)
            except Exception as e:
                print('Failed to run test:', e)
            break
    time.sleep(poll)
    elapsed += poll

print('Watcher exiting')
