from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
import os

import threading
import json
from process_upload import process, read_global_kb
from edge_sync import run_poll
import threading

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / 'pipeline' / 'uploads'
OUTPUT_DIR = BASE_DIR / 'pipeline' / 'output'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    f = request.files['file']
    if f.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(f.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    saved_path = UPLOAD_DIR / unique_name
    f.save(saved_path)

    # Start background processing so UI can poll status
    jobid = uuid.uuid4().hex
    status_path = OUTPUT_DIR / f"{jobid}.status.json"
    summary_path = OUTPUT_DIR / f"{jobid}.summary.json"

    def background():
        try:
            summary = process(saved_path, status_path=status_path)
            # save summary
            try:
                summary_path.write_text(json.dumps(summary, ensure_ascii=False))
            except Exception:
                pass
        except Exception as e:
            _err = {"stage": "error", "message": str(e)}
            try:
                status_path.write_text(json.dumps(_err, ensure_ascii=False))
            except Exception:
                pass

    t = threading.Thread(target=background, daemon=True)
    t.start()

    return redirect(url_for('status_page', jobid=jobid))


@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route('/status/<jobid>')
def status(jobid):
    status_path = OUTPUT_DIR / f"{jobid}.status.json"
    if status_path.exists():
        try:
            return status_path.read_text(encoding='utf-8')
        except Exception:
            return json.dumps({"stage": "unknown", "message": "could not read status"})
    else:
        return json.dumps({"stage": "pending", "message": "waiting for processing to start"})


@app.route('/status_page/<jobid>')
def status_page(jobid):
    return render_template('status.html', jobid=jobid)


@app.route('/summary/<jobid>')
def summary(jobid):
    summary_path = OUTPUT_DIR / f"{jobid}.summary.json"
    if summary_path.exists():
        try:
            return summary_path.read_text(encoding='utf-8'), 200, {'Content-Type': 'application/json'}
        except Exception:
            return json.dumps({}), 500
    return json.dumps({}), 404


@app.route('/chat_page')
def chat_page():
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    q = (data.get('q') or '').strip()
    if not q:
        return {'reply': '', 'source': ''}

    # load global KB entries
    try:
        entries = read_global_kb(BASE_DIR / 'global_kb.jsonl')
    except Exception:
        entries = []

    if not entries:
        return {'reply': 'No dataset available. Please upload data first.', 'source': ''}

    # Retrieve relevant entries using sequence matching (get top 5 for context)
    from difflib import SequenceMatcher
    scored_entries = []
    qnorm = q.lower()
    for e in entries:
        combined = ((e.get('topic') or '') + ' ' + (e.get('text') or '')).lower()
        score = SequenceMatcher(None, qnorm, combined).ratio()
        if score > 0.1:  # Lower threshold to get more context
            scored_entries.append((score, e))
    
    # Sort by score and take top 5
    scored_entries.sort(reverse=True, key=lambda x: x[0])
    relevant_entries = [e for _, e in scored_entries[:5]]

    if not relevant_entries:
        return {'reply': 'No relevant entries found in the dataset for this question.', 'source': ''}

    # Build source information from best matches
    best_entry = relevant_entries[0]
    source = f"id:{best_entry.get('id', '')} topic:{best_entry.get('topic','')}"
    if len(relevant_entries) > 1:
        source += f" (+{len(relevant_entries)-1} more)"

    # Use TinyLlama model to generate answer based on dataset context
    try:
        from llama_local import answer_with_model
        reply = answer_with_model(q, relevant_entries)
        if reply:
            return {'reply': reply, 'source': source}
    except Exception as e:
        # If model fails, fall back to best match text
        import traceback
        print(f"Model error: {e}")
        traceback.print_exc()
        pass

    # Fallback: return best match text if model fails
    reply = best_entry.get('text') or 'Unable to generate answer. Please try rephrasing your question.'
    return {'reply': reply, 'source': source}


if __name__ == '__main__':
    # start edge sync worker in background
    t = threading.Thread(target=run_poll, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=True)
