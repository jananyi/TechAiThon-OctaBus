# ADMIN → SYSTEM PROCESS pipeline

This small pipeline implements the steps you described using Python. It expects the workspace root to contain `global_kb.jsonl` (provided).

Steps implemented:
- Admin uploads Excel/CSV → `process_upload.py` reads it
- System assigns IDs continuing from the maximum ID found in `global_kb.jsonl` (starts after 200)
- Edge layer: converts to JSONL, encrypts payload (`edge.enc`) and stores `edge.key`
- Fog layer: decrypts edge payload, checks duplicates against `global_kb.jsonl`, reports counts, encrypts new entries (`fog.enc`) and stores `fog.key`
- Global layer: decrypts fog payload and appends only new entries to `global_kb.jsonl`, creates `global.enc` and `global.key`

Output files (in `pipeline/output/`):
- `edge.enc`, `edge.key`
- `fog.enc`, `fog.key`
- `global.enc`, `global.key`
- `edge_payload.jsonl`, `fog_payload.jsonl`, `global_kb_updated.jsonl` (intermediate/plain files)

Requirements:
 - Create a virtualenv and install packages from `requirements.txt`:

```bash
python -m venv .venv
.
# Windows
.venv\Scripts\activate
pip install -r pipeline/requirements.txt
```

Usage:

```bash
python pipeline/process_upload.py --input path/to/admin_upload.xlsx
```

Notes:
- The script deduplicates entries by matching `topic` + `text` (case-insensitive, trimmed).
- IDs are numeric and continue from the maximum existing `id` in `global_kb.jsonl` (which in the provided file ends at 200), so new IDs start at 201.
- Keys are Fernet symmetric keys stored in `pipeline/output/`.
