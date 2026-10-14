"""Fresh 1,000 x 9 experiment with no client maxOutputTokens setting.
"""
import argparse
from collections import Counter
import concurrent.futures as futures
import csv
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_gemini import parse_response, digest, save, BASE

CONDITIONS = {'standard', 'graph_aware_standard', 'cot_no_graph', 'cot_graph',
              'contrastive_cot_no_graph', 'contrastive_cot_graph',
              'standard_parent_only', 'standard_siblings_only', 'standard_children_only'}


def valid(data, condition):
    if data.get('status') != 'completed':
        return False
    status, text, predictions, explanation, finish = parse_response(data.get('raw_response', {}))
    return (status == 'completed' and bool(predictions)
            and predictions == data.get('predictions')
            and ('cot' not in condition or bool(explanation.strip())))


def process_alive(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=,command='],
                            capture_output=True, text=True)
    text = result.stdout.strip()
    return text if text and not text.startswith('Z') else ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='results/external_demo/gemini')
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--rpm', type=float, default=120)
    p.add_argument('--max-jobs', type=int)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if a.workers < 1 or a.rpm <= 0 or (a.max_jobs is not None and a.max_jobs < 1):
        p.error('workers, rpm and max-jobs must be positive')
    out = ROOT / a.output
    cfg = dict(model='gemini-3.8-flash', temperature=1.0,
               max_output_tokens=None, thinking='provider default; no explicit override',
               response_format='JSON MIME, prompt-defined schema', seed_policy='Omitted; provider generates seed')
    lock = out / 'RUNNING.lock'
    if lock.exists() and not (a.dry_run or a.verify_only):
        if process_alive(int(lock.read_text())):
            raise SystemExit('Fresh run is already running. Do not start a second copy.')
        lock.unlink(missing_ok=True)
    cfg.update(output_limit_policy='Provider default; maxOutputTokens omitted',
               runner_sha256=digest(Path(__file__).read_text()),
               manifest_sha256=digest((ROOT/'prompts_v6/all_1k/manifest.csv').read_text()))
    config_path = out / 'run_config.json'
    if not config_path.exists() and any(out.glob('example_*/*_result.json')):
        raise SystemExit('Results exist without a configuration. Refusing to mix results.')

    manifest = ROOT / 'prompts_v6/all_1k/manifest.csv'
    rows = list(csv.DictReader(manifest.open()))
    pairs = {(r['entity_id'], r['condition']) for r in rows}
    entities = {r['entity_id'] for r in rows}
    if len(rows) != 9000 or len(pairs) != 9000 or len(entities) != 1000:
        raise SystemExit('Expected exactly 1000 entities and 9000 unique pairs.')
    if pairs != {(e, c) for e in entities for c in CONDITIONS}:
        raise SystemExit('Manifest does not contain precisely the nine required conditions for every entity.')
    jobs, fingerprints, expected = [], [], set()
    for row in rows:
        prompt = (ROOT / row['prompt_path']).read_text()
        meta = json.loads((ROOT / row['metadata_path']).read_text())
        if meta['entity_id'] != row['entity_id']:
            raise SystemExit('Metadata identity mismatch')
        fp = digest(prompt + json.dumps(meta, sort_keys=True))
        fingerprints.append(fp)
        path = out / f"example_{int(row['example_number']):04d}" / f"{row['condition']}_result.json"
        expected.add(path)
        data = json.loads(path.read_text()) if path.exists() else {}
        if data and (data.get('input_hash') != fp or data.get('entity_id') != row['entity_id']
                     or data.get('condition') != row['condition'] or data.get('model') != cfg['model']):
            raise SystemExit(f'Existing result identity/config mismatch: {path}')
        jobs.append((row, fp, path, prompt, meta, data))
    cfg['inputs_sha256'] = digest(''.join(fingerprints))
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise SystemExit('Inputs/configuration changed; use a new output directory.')
    if set(out.glob('example_*/*_result.json')) - expected:
        raise SystemExit('Unexpected final result files found. Resolve these before recovery.')

    def audit(write=False):
        counts = Counter()
        unresolved = []
        for row, fp, path, prompt, meta, old in jobs:
            data = json.loads(path.read_text()) if path.exists() else {}
            if valid(data, row['condition']):
                counts['completed'] += 1
            else:
                counts[data.get('status', 'missing')] += 1
                unresolved.append({'entity_id': row['entity_id'], 'condition': row['condition'],
                                   'status': data.get('status', 'missing'),
                                   'reason': data.get('finish_reason', data.get('http_status'))})
        report = {'required': 9000, 'entities': 1000, 'counts': dict(counts),
                  'complete': counts['completed'] == 9000, 'unresolved': unresolved}
        if write:
            save(out / 'recovery_completion.json', report)
        print(f"Validated: {counts['completed']}/9000; unresolved: {len(unresolved)}", flush=True)
        return report

    report = audit()
    if a.verify_only:
        return 0 if report['complete'] else 2
    pending = [j for j in jobs if not valid(j[-1], j[0]['condition'])]
    pending.sort(key=lambda j: not bool(j[-1]))  # Repair saved failures before new jobs.
    if a.max_jobs:
        pending = pending[:a.max_jobs]
    print(f'This invocation: {len(pending)} jobs; completed results will not be overwritten.', flush=True)
    if a.dry_run or not pending:
        return 0
    key = os.environ.get('GEMINI_API_KEY') or getpass.getpass('Gemini API key (hidden): ')
    if not key:
        raise SystemExit('No API key supplied')

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(BASE + path, data=data,
                headers={'x-goog-api-key': key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    model_path = 'models/' + urllib.parse.quote(cfg['model'], safe='')
    info = request(model_path)
    if 'generateContent' not in info.get('supportedGenerationMethods', []):
        raise SystemExit('Model does not support generateContent.')
    print('Client output cap: omitted. Model outputTokenLimit:', info.get('outputTokenLimit'), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    stop = threading.Event()
    gate = threading.Lock()
    clock = [0.0]
    run_id = uuid.uuid4().hex

    def interrupt(signum, frame):
        print('\nStopping new requests; saving in-flight responses (up to 180 seconds).', flush=True)
        stop.set()

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def pace():
        with gate:
            now = time.monotonic()
            start = max(now, clock[0])
            clock[0] = start + 60 / a.rpm
        return not stop.wait(max(0, start - time.monotonic()))

    def run(job):
        row, fp, path, prompt, meta, old = job
        if old:
            save(out / 'attempts' / f'recovery_prior_{run_id}_{path.parent.name}_{row["condition"]}.json', old)
        transient = 0
        malformed = 0
        while not stop.is_set():
            if not pace():
                return
            generation = {'temperature': cfg['temperature'],
                          'responseMimeType': 'application/json'}
            record = dict(row, example_number=int(row['example_number']), input_hash=fp,
                          model=cfg['model'], prompt=prompt, metadata=meta, predictions=[],
                          status='api_error', generation_config=generation, recovery_run=run_id,
                          timestamp_utc=datetime.now(timezone.utc).isoformat())
            retry = False
            try:
                raw = request(model_path + ':generateContent', {
                    'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                    'generationConfig': generation})
                status, text, pred, exp, finish = parse_response(raw)
                if status == 'completed' and (not pred or ('cot' in row['condition'] and not exp.strip())):
                    status = 'parse_error'
                record.update(status=status, response=text, predictions=pred, explanation=exp,
                              finish_reason=finish, raw_response=raw,
                              model_version=raw.get('modelVersion'), usage=raw.get('usageMetadata', {}))
                if status in ('incomplete', 'parse_error') and finish in ('STOP', 'MAX_TOKENS') and malformed < 1:
                    malformed += 1
                    retry = True
            except urllib.error.HTTPError as ex:
                record.update(http_status=ex.code, error=f'HTTP {ex.code}')
                transient += 1
                retry = ex.code in (429, 500, 502, 503, 504) and transient < 4
            except (urllib.error.URLError, TimeoutError) as ex:
                record['error'] = type(ex).__name__
                transient += 1
                retry = transient < 4
            save(out / 'attempts' / f'recovery_{run_id}_{uuid.uuid4().hex}.json', record)
            save(path, record)
            print(f"{row['example_number']} {row['condition']}: {record['status']} "
                  f"reason={record.get('finish_reason', record.get('http_status'))} "
                  "output_cap=provider_default" + ('; retrying' if retry else ''), flush=True)
            if record['status'] == 'completed':
                return
            if not retry:
                stop.set()
                print('UNRESOLVED RESPONSE: stopping new work. Saved results retained.', flush=True)
                return
            if transient:
                stop.wait(min(60, 2 ** transient))

    try:
        save(config_path, cfg)
        save(out / 'model_info.json', info)
        save(out / 'recovery_runs' / f'{run_id}.json', {
            'original_config': cfg, 'runner_sha256': digest(Path(__file__).read_text()),
            'client_token_cap': None, 'reason_for_change': 'External WiKC demonstration; fresh results only',
            'policy': 'Fresh run; retain first valid response; retry truncation/parse failure once; no score-based selection.',
            'reference': 'https://ai.google.dev/gemini-api/docs/generate-content/thinking',
            'created_utc': datetime.now(timezone.utc).isoformat()})
        with futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            iterator = iter(pending)
            active = set()
            for _ in range(min(a.workers, len(pending))):
                active.add(pool.submit(run, next(iterator)))
            while active:
                done, active = futures.wait(active, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        future.result()
                    except Exception:
                        stop.set()
                        raise
                    if not stop.is_set():
                        job = next(iterator, None)
                        if job is not None:
                            active.add(pool.submit(run, job))
        report = audit(write=True)
        if report['complete']:
            print('COMPLETE: 1000 entities x 9 conditions = 9000 validated results.')
            return 0
        print('NOT COMPLETE. Do not generate final metrics yet.')
        return 2 if stop.is_set() or not a.max_jobs else 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.exit(main())