"""Pipeline script to process an admin-uploaded Excel/CSV into JSONL and run edge/fog/global steps.

Usage:
  python process_upload.py --input path/to/file.xlsx

This script will:
 - Read the Excel/CSV file
 - Assign IDs continuing from max ID in global_kb.jsonl (starts from 201)
 - Edge: encrypt the converted JSONL payload and save key/file
 - Fog: decrypt edge payload, check duplicates vs global_kb.jsonl, report counts, encrypt new entries
 - Global: decrypt fog payload and append new entries to global_kb.jsonl, then encrypt updated file

Output files are written to ./pipeline/output/
"""
import argparse
import json
import os
from pathlib import Path
from collections import OrderedDict
import sys
import threading
import time
import pickle
import io
import torch
from datetime import datetime
from typing import Callable

import pandas as pd

from encrypt_utils import generate_key, load_key, encrypt_bytes, decrypt_bytes, encrypt_and_write


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_KB = WORKSPACE_ROOT / "global_kb.jsonl"
OUTPUT_DIR = WORKSPACE_ROOT / "pipeline" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUTPUT_DIR / 'global_index.pkl'
EDGE_STORE = WORKSPACE_ROOT / 'pipeline' / 'edge_store'
EDGE_STORE.mkdir(parents=True, exist_ok=True)


def read_input(path: Path):
    if path.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        # Try reading CSV with different separators (comma, semicolon, tab)
        # Pick the separator that gives the most columns (typically the correct one)
        best_df = None
        best_cols = 0
        for sep in [',', ';', '\t']:
            try:
                test_df = pd.read_csv(path, sep=sep)
                # If we get multiple columns, this is likely the correct separator
                if len(test_df.columns) > best_cols:
                    best_df = test_df
                    best_cols = len(test_df.columns)
            except Exception:
                continue
        if best_df is not None and best_cols > 1:
            df = best_df
        else:
            # Fallback: try comma (default)
            df = pd.read_csv(path)
    return df


def normalize_column_name(name: str) -> str:
    return name.strip().lower()


def df_to_records(df: pd.DataFrame):
    # Map expected columns flexibly
    colmap = {normalize_column_name(c): c for c in df.columns}
    def get(col):
        key = normalize_column_name(col)
        return df[colmap[key]] if key in colmap else None

    # Build list of dicts
    records = []
    for _, row in df.iterrows():
        def val(key):
            if key not in colmap:
                return ''
            col_name = colmap[key]  # Get the actual column name
            try:
                v = row[col_name]  # Access pandas Series by column name
            except (KeyError, IndexError):
                return ''
            if pd.isna(v):
                return ''
            return str(v).strip()

        rec = {
            "topic": val('topic'),
            "text": val('text'),
            "treatment date": val('treatment date'),
            "treatment country": val('treatment country'),
            "treatment given": val('treatment given'),
        }
        records.append(rec)
    return records


def validate_and_normalize_record(rec: dict):
    """Validate and normalize a single record. Returns normalized record or raises ValueError."""
    # Required fields: topic, text
    topic = (rec.get('topic') or '').strip()
    text = (rec.get('text') or '').strip()
    if not topic:
        raise ValueError('Missing topic')
    if not text:
        raise ValueError('Missing text')

    # Normalize date to YYYY-MM-DD if possible
    date_val = rec.get('treatment date') or ''
    date_str = ''
    if date_val:
        try:
            # allow pandas/Excel style datetimes or strings
            dt = pd.to_datetime(date_val)
            date_str = dt.strftime('%Y-%m-%d')
        except Exception:
            # fallback: try parse with datetime
            try:
                dt = datetime.fromisoformat(str(date_val))
                date_str = dt.strftime('%Y-%m-%d')
            except Exception:
                date_str = str(date_val)

    norm = {
        'topic': topic,
        'text': text,
        'treatment date': date_str,
        'treatment country': (rec.get('treatment country') or '').strip(),
        'treatment given': (rec.get('treatment given') or '').strip(),
    }
    return norm


def read_global_kb(path: Path):
    entries = []
    if not path.exists():
        return entries
    with path.open('r', encoding='utf-8') as f:
        decoder = json.JSONDecoder()
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            idx = 0
            length = len(s)
            # handle possibility of multiple JSON objects concatenated on the same line
            while idx < length:
                try:
                    obj, used = decoder.raw_decode(s[idx:])
                    entries.append(obj)
                    idx += used
                    # skip whitespace between objects
                    while idx < length and s[idx].isspace():
                        idx += 1
                except Exception:
                    # if we can't decode remaining text, stop attempting to parse this line
                    break
    return entries


def build_global_index(path: Path):
    """Build a set of (topic,text) keys from global_kb.jsonl and persist it for fast checks."""
    keys = set()
    if not path.exists():
        return keys
    decoder = json.JSONDecoder()
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            idx = 0
            length = len(s)
            while idx < length:
                try:
                    e, used = decoder.raw_decode(s[idx:])
                    topic = (e.get('topic') or '').strip().lower()
                    text = (e.get('text') or '').strip().lower()
                    keys.add((topic, text))
                    idx += used
                    while idx < length and s[idx].isspace():
                        idx += 1
                except Exception:
                    break
    # persist index
    try:
        with INDEX_PATH.open('wb') as fh:
            pickle.dump(keys, fh)
    except Exception:
        pass
    return keys


def load_global_index():
    if INDEX_PATH.exists():
        try:
            with INDEX_PATH.open('rb') as fh:
                return pickle.load(fh)
        except Exception:
            return build_global_index(GLOBAL_KB)
    else:
        return build_global_index(GLOBAL_KB)

def _write_status(status_path: Path, payload: dict):
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


def write_jsonl(path: Path, records):
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_to_global(path: Path, records):
    # Ensure file ends with a newline before appending to avoid concatenating objects
    if path.exists():
        try:
            with path.open('rb') as fh:
                fh.seek(-1, os.SEEK_END)
                last = fh.read(1)
                needs_nl = last != b"\n"
        except Exception:
            needs_nl = False
        if needs_nl:
            with path.open('a', encoding='utf-8') as f:
                f.write('\n')

    with path.open('a', encoding='utf-8') as f:
        for r in records:
            # Ensure output has keys in the required order: id, topic, text, treatment date, treatment country, treatment given
            out = OrderedDict()
            # id might be int or str
            out['id'] = int(r.get('id')) if r.get('id') is not None else None
            out['topic'] = r.get('topic', '')
            out['text'] = r.get('text', '')
            out['treatment date'] = r.get('treatment date', '')
            out['treatment country'] = r.get('treatment country', '')
            out['treatment given'] = r.get('treatment given', '')
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def find_max_id(entries):
    max_id = 200
    for e in entries:
        try:
            iid = int(e.get('id', 0))
            if iid > max_id:
                max_id = iid
        except Exception:
            continue
    return max_id


def deduplicate_against_global(records, global_entries=None, index_keys=None):
    # Prefer using a persisted index for speed; fallback to scanning global entries
    if index_keys is None:
        index_keys = load_global_index()

    new = []
    dup = []
    seen = set(index_keys)
    for r in records:
        key = (r.get('topic','').strip().lower(), r.get('text','').strip().lower())
        if key in seen:
            dup.append(r)
        else:
            new.append(r)
            seen.add(key)
    return dup, new


def process(input_path: Path, status_path: Path = None):
    df = read_input(input_path)
    records = df_to_records(df)

    # Read global KB
    global_entries = read_global_kb(GLOBAL_KB)
    max_id = find_max_id(global_entries)
    next_id = max_id + 1

    # Assign IDs
    for r in records:
        r['id'] = next_id
        next_id += 1
    if status_path:
        _write_status(status_path, {"stage": "ids_assigned", "message": f"Assigned IDs starting at {max_id+1}", "count": len(records)})
    # Validate/normalize records
    normalized = []
    for r in records:
        try:
            nr = validate_and_normalize_record(r)
            # keep id
            nr['id'] = int(r['id'])
            normalized.append(nr)
        except Exception as e:
            # skip invalid records
            continue
    records = normalized

    # EDGE: create torch .pt payload in-memory and encrypt
    if status_path:
        _write_status(status_path, {"stage": "edge_start", "message": "Edge: converting and encrypting payload"})
    time.sleep(1)  # Delay to show animation
    # write human-readable edge payload JSONL for inspection/debug
    try:
        write_jsonl(OUTPUT_DIR / 'edge_payload.jsonl', records)
    except Exception:
        pass
    edge_key_path = OUTPUT_DIR / 'edge.key'
    edge_pt_enc = OUTPUT_DIR / 'edge.pt'
    buf = io.BytesIO()
    # torch.save will serialize Python objects into a bytes buffer
    torch.save(records, buf)
    edge_plain_bytes = buf.getvalue()
    encrypt_and_write(edge_plain_bytes, edge_pt_enc, edge_key_path)
    time.sleep(1)  # Delay after encryption
    if status_path:
        _write_status(status_path, {
            "stage": "edge_done",
            "message": "Edge: payload encrypted",
            "edge_enc": str(edge_pt_enc),
            "steps": {
                "edge_updated": True,
                "fog_validated": False,
                "validated_dataset": False,
                "stored_in_global": False,
                "encrypted_global": False,
                "synced_to_edge": False,
            }
        })

    # FOG: decrypt edge, dedupe vs global (use index for speed)
    if status_path:
        _write_status(status_path, {"stage": "fog_start", "message": "Fog: decrypting edge payload and validating against global index"})
    time.sleep(1)  # Delay to show animation
    edge_plain = decrypt_bytes(edge_pt_enc.read_bytes(), load_key(edge_key_path))
    edge_records = torch.load(io.BytesIO(edge_plain))

    # Load or build index (fast repeated checks)
    index_keys = load_global_index()
    duplicates, new_entries = deduplicate_against_global(edge_records, index_keys=index_keys)
    time.sleep(1)  # Delay after validation

    fog_report = f"Found {len(duplicates)} duplicates, {len(new_entries)} new entries"
    if status_path:
        # update fog validation step
        _write_status(status_path, {
            "stage": "fog_done",
            "message": fog_report,
            "duplicates": len(duplicates),
            "new_entries": len(new_entries),
            "steps": {
                "edge_updated": True,
                "fog_validated": True,
                "validated_dataset": False,
                "stored_in_global": False,
                "encrypted_global": False,
                "synced_to_edge": False,
            }
        })

    # Save fog pickled payload (only new entries) and encrypt
    fog_key_path = OUTPUT_DIR / 'fog.key'
    fog_pt_enc = OUTPUT_DIR / 'fog.pt'
    # write human-readable fog payload for inspection/debug
    try:
        write_jsonl(OUTPUT_DIR / 'fog_payload.jsonl', new_entries)
    except Exception:
        pass
    buf2 = io.BytesIO()
    torch.save(new_entries, buf2)
    fog_plain_bytes = buf2.getvalue()
    encrypt_and_write(fog_plain_bytes, fog_pt_enc, fog_key_path)
    time.sleep(1)  # Delay after fog encryption
    if status_path:
        _write_status(status_path, {"stage": "fog_saved", "message": "Fog: new entries encrypted", "fog_enc": str(fog_pt_enc),
            "steps": {
                "edge_updated": True,
                "fog_validated": True,
                "validated_dataset": True,
                "stored_in_global": False,
                "encrypted_global": False,
                "synced_to_edge": False,
            }
        })

    # GLOBAL: decrypt fog payload and append new entries to global_kb.jsonl
    fog_plain = decrypt_bytes(fog_pt_enc.read_bytes(), load_key(fog_key_path))
    fog_new = torch.load(io.BytesIO(fog_plain))

    if fog_new:
        if status_path:
            _write_status(status_path, {"stage": "global_updating", "message": "Global: appending new entries to global_kb.jsonl",
                "steps": {
                    "edge_updated": True,
                    "fog_validated": True,
                    "validated_dataset": True,
                    "stored_in_global": True,
                    "encrypted_global": False,
                    "synced_to_edge": False,
                }
            })
        time.sleep(1)  # Delay to show animation
        append_to_global(GLOBAL_KB, fog_new)
        # rebuild index (since global changed)
        index_keys = build_global_index(GLOBAL_KB)
        # write a human-readable snapshot of the updated global KB for inspection
        try:
            write_jsonl(OUTPUT_DIR / 'global_kb_updated.jsonl', read_global_kb(GLOBAL_KB))
        except Exception:
            pass
        time.sleep(1)  # Delay after appending

    # Create encrypted global .pt from the updated global_kb
    if status_path:
        _write_status(status_path, {"stage": "global_pt", "message": "Global: creating encrypted global .pt",
            "steps": {
                "edge_updated": True,
                "fog_validated": True,
                "validated_dataset": True,
                "stored_in_global": True,
                "encrypted_global": True,
                "synced_to_edge": False,
            }
        })
    time.sleep(1)  # Delay to show animation
    all_global = read_global_kb(GLOBAL_KB)
    buf3 = io.BytesIO()
    torch.save(all_global, buf3)
    global_plain_bytes = buf3.getvalue()
    global_key_path = OUTPUT_DIR / 'global.key'
    global_pt_enc = OUTPUT_DIR / 'global.pt'
    encrypt_and_write(global_plain_bytes, global_pt_enc, global_key_path)
    time.sleep(1)  # Delay after global encryption

    # Touch a sync marker so edge layer can pick up new global
    sync_marker = OUTPUT_DIR / 'global_sync.flag'
    sync_marker.write_text(str(time.time()))
    # Wait briefly for edge sync to pick up the new global (edge_sync may be running separately)
    synced = False
    try:
        wait_seconds = 5
        for _ in range(wait_seconds * 2):
            if (EDGE_STORE / 'sync.flag').exists():
                synced = True
                break
            time.sleep(0.5)
    except Exception:
        synced = False

    if status_path:
        _write_status(status_path, {"stage": "done", "message": "All done", "global_enc": str(global_pt_enc),
            "steps": {
                "edge_updated": True,
                "fog_validated": True,
                "validated_dataset": True,
                "stored_in_global": True,
                "encrypted_global": True,
                "synced_to_edge": bool(synced),
            }
        })

    summary = {
        'report': fog_report,
        'duplicates': len(duplicates),
        'new_entries': len(new_entries),
        'edge_enc': str(edge_pt_enc.resolve()),
        'fog_enc': str(fog_pt_enc.resolve()),
        'global_enc': str(global_pt_enc.resolve()),
        'edge_key': str(edge_key_path.resolve()),
        'fog_key': str(fog_key_path.resolve()),
        'global_key': str(global_key_path.resolve()),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to Excel (.xlsx) or CSV file')
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print('Input file not found:', input_path)
        sys.exit(1)
    process(input_path)


if __name__ == '__main__':
    main()
