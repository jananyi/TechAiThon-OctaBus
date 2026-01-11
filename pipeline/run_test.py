from pathlib import Path
import uuid
import csv
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.process_upload import process
TEST_CSV = ROOT / 'pipeline' / 'test_dup.csv'
OUTPUT = ROOT / 'pipeline' / 'output'

rows = [
    {'topic':'rs','text':'djd','treatment date':'2026-01-10','treatment country':'cdc','treatment given':'xd'},
    {'topic':'rs','text':'djd','treatment date':'2026-01-10','treatment country':'cdc','treatment given':'xd'},
]

TEST_CSV.parent.mkdir(parents=True, exist_ok=True)
with TEST_CSV.open('w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=['topic','text','treatment date','treatment country','treatment given'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

jobid = uuid.uuid4().hex
status_path = OUTPUT / f'{jobid}.status.json'
print('Running process on', TEST_CSV)
summary = process(TEST_CSV, status_path=status_path)
print('Summary:', summary)
