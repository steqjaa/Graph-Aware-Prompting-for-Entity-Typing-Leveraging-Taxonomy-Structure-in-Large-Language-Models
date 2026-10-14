"""Verified, ordered loading of immutable final responses."""
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def prepare_resources(root):
    spec = json.loads((root / 'resources/manifest.json').read_text())
    for item in spec['resources']:
        target = root / item['path']
        if target.exists() and digest(target) == item['sha256']:
            continue
        source = root / item['compressed_path']
        if digest(source) != item['compressed_sha256']:
            raise ValueError(f'Corrupt compressed resource: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.restoring')
        with gzip.open(source, 'rb') as src, temp.open('wb') as out:
            shutil.copyfileobj(src, out)
        if digest(temp) != item['sha256']:
            temp.unlink()
            raise ValueError(f'Restored resource checksum mismatch: {target}')
        temp.replace(target)

def load_records(root):
    records, attempts, configs = {}, [], {}
    source_manifest = root / 'configs/raw_files.jsonl'
    for line in source_manifest.read_text().splitlines():
        item = json.loads(line)
        path = root / item['path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError(f'Raw evidence changed: {item["path"]}')
        value = json.loads(raw)
        run = item['run']
        if item['kind'] == 'result':
            key = run, int(value['example_number']), value['condition']
            if key in records:
                raise ValueError(f'Duplicate result: {key}')
            records[key] = value
        elif item['kind'] == 'attempt':
            value['_run'] = run
            attempts.append(value)
        else:
            configs[run + ' ' + item['path']] = value
    return records, attempts, configs

def validate_records(root, records, runs=None):
    """Validate scheduled pairs, prompt hashes, metadata, response parsing and seeds."""
    from .validation import parse_response
    jobs = {}
    for part in ('all_1k', 'children_removed'):
        with (root / f'prompts_v6/{part}/manifest.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            prompt = (root / row['prompt_path']).read_text()
            meta = json.loads((root / row['metadata_path']).read_text())
            key = int(row['example_number']), row['condition']
            jobs[key] = (row, prompt, meta)
    expected = set()
    summary = []
    for run in (runs or ('Gemini', 'Llama 0', 'Llama 42', 'Llama 123')):
        counts = {'run': run, 'scheduled': 0, 'completed': 0, 'failed': 0}
        for key, (row, prompt, meta) in jobs.items():
            child = key[1].endswith('_no_children')
            if child and run != 'Llama 0':
                continue
            full_key = (run, *key)
            expected.add(full_key)
            if full_key not in records:
                raise ValueError(f'Missing scheduled record: {full_key}')
            value = records[full_key]
            fp = hashlib.sha256((prompt + json.dumps(meta, sort_keys=True)).encode()).hexdigest()
            if (value.get('entity_id') != row['entity_id'] or value.get('prompt') != prompt
                    or value.get('metadata') != meta or value.get('input_hash') != fp):
                raise ValueError(f'Input identity mismatch: {full_key}')
            model = 'gemini-3.8-flash' if run == 'Gemini' else 'meta-llama/Llama-3.1-8B-Instruct'
            if value.get('model') != model:
                raise ValueError(f'Model mismatch: {full_key}')
            if run != 'Gemini':
                if value.get('generation_config', {}).get('seed') != int(run.split()[1]):
                    raise ValueError(f'Seed mismatch: {full_key}')
                if value.get('revision') != '0e9e39f249a16976918f6564b8830bc894c89659':
                    raise ValueError(f'Checkpoint mismatch: {full_key}')
            if value.get('status') == 'completed':
                state, pred, explanation = parse_response(value.get('response', ''), key[1], value.get('finish_reason') == 'STOP')
                if state != 'completed' or pred != value.get('predictions') or explanation != value.get('explanation', ''):
                    raise ValueError(f'Completed response parsing mismatch: {full_key}')
                counts['completed'] += 1
            else:
                counts['failed'] += 1
            counts['scheduled'] += 1
        summary.append(counts)
    if set(records) != expected:
        raise ValueError('Unexpected results outside the final study')
    return summary
