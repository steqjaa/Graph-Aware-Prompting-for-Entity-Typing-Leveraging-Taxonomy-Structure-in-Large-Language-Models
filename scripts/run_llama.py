import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temp.replace(path)

def parse(text, condition, stopped=True):
    if not stopped:
        return 'incomplete', [], ''
    try:
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
        obj = json.loads(clean)
        pred, explanation = obj['types'], obj.get('explanation', '')
        if not isinstance(pred, list) or not pred or any(not isinstance(x, str) or not x.strip() for x in pred):
            raise ValueError('Invalid types')
        if not isinstance(explanation, str) or ('cot' in condition and not explanation.strip()):
            raise ValueError('Missing explanation')
        return 'completed', pred, explanation
    except (ValueError, KeyError, TypeError):
        return 'parse_error', [], ''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='prompts_v5/all_1k/manifest.csv')
    p.add_argument('--output', default='results/llama_1k')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-jobs', type=int)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--retry-failed', action='store_true')
    a = p.parse_args()
    if a.batch_size < 1 or (a.max_jobs is not None and a.max_jobs < 1):
        p.error('batch-size and max-jobs must be positive')
    manifest, out = ROOT / a.manifest, ROOT / a.output
    rows = list(csv.DictReader(manifest.open()))
    if not rows:
        p.error('Empty manifest')
    jobs, seen, paths = [], set(), set()
    for row in rows:
        key = row['entity_id'], row['condition']
        path = out / f"example_{int(row['example_number']):04d}" / f"{row['condition']}_result.json"
        if key in seen or path in paths:
            p.error('Duplicate manifest entity/condition or output path')
        seen.add(key); paths.add(path)
        prompt = (ROOT / row['prompt_path']).read_text()
        meta = json.loads((ROOT / row['metadata_path']).read_text())
        if meta['entity_id'] != row['entity_id']:
            p.error('Metadata identity mismatch')
        fp = digest(prompt + json.dumps(meta, sort_keys=True))
        jobs.append((row, path, prompt, meta, fp))
    cfg = dict(model=MODEL, revision=REVISION, dtype='bfloat16', do_sample=False,
               max_new_tokens=800, batch_size=a.batch_size,
               manifest_sha256=digest(manifest.read_text()),
               inputs_sha256=digest(''.join(j[4] for j in jobs)),
               runner_sha256=digest(Path(__file__).read_text()))
    cp = out / 'run_config.json'
    if cp.exists() and json.loads(cp.read_text()) != cfg:
        p.error('Configuration or inputs changed; use a new output directory')
    if not cp.exists() and list(out.glob('example_*/*_result.json')):
        p.error('Results without run configuration')
    if set(out.glob('example_*/*_result.json')) - paths:
        p.error('Unexpected result files')
    def audit():
        pending, counts = [], dict(completed=0, failed=0, missing=0)
        for job in jobs:
            row, path, _, _, fp = job
            if not path.exists():
                counts['missing'] += 1; pending.append(job); continue
            try:
                x = json.loads(path.read_text())
            except ValueError:
                p.error(f'Corrupt result: {path}')
            if any(x.get(k) != v for k, v in dict(input_hash=fp, entity_id=row['entity_id'], condition=row['condition'], model=MODEL, revision=REVISION).items()):
                p.error(f'Result identity mismatch: {path}')
            status, pred, exp = parse(x.get('response', ''), row['condition'], x.get('finish_reason') == 'STOP')
            valid = x.get('status') == status == 'completed' and x.get('predictions') == pred and x.get('explanation') == exp
            counts['completed' if valid else 'failed'] += 1
            if not valid and a.retry_failed:
                pending.append(job)
        print(f"Completed {counts['completed']}/{len(jobs)} | failed {counts['failed']} | missing {counts['missing']}", flush=True)
        return pending, counts
    pending, counts = audit()
    if a.verify_only:
        return 0 if counts['completed'] == len(jobs) else 2
    if a.max_jobs:
        pending = pending[:a.max_jobs]
    if a.dry_run or not pending:
        print(f'Pending this invocation: {len(pending)}'); return 0
    out.mkdir(parents=True, exist_ok=True)
    # flock is released by the OS even after an interrupted process.
    import fcntl
    with (out / 'RUNNING.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another runner owns this output directory')
        if cp.exists() and json.loads(cp.read_text()) != cfg:
            p.error('Configuration changed while acquiring the run lock')
        pending, _ = audit()
        if a.max_jobs:
            pending = pending[:a.max_jobs]
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            p.error('A BF16-capable CUDA GPU is required')
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16).to('cuda').eval()
        stops = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')})
        environment = dict(torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name())
        environment_path = out / 'environment.json'
        if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
            p.error('Runtime versions/GPU changed; use the original environment or a new output directory')
        save(cp, cfg)
        save(environment_path, environment)
        for start in range(0, len(pending), a.batch_size):
            batch = pending[start:start + a.batch_size]
            texts = [tokenizer.apply_chat_template([{'role':'user', 'content':j[2]}], tokenize=False, add_generation_prompt=True) for j in batch]
            inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
            if inputs.input_ids.shape[1] + 800 > model.config.max_position_embeddings:
                p.error('Prompt exceeds context window; refusing truncation')
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=800, do_sample=False,
                    num_beams=1, eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
            for job, tokens in zip(batch, generated[:, inputs.input_ids.shape[1]:].tolist()):
                row, path, prompt, meta, fp = job
                end = next((i for i, token in enumerate(tokens) if token in stops), None)
                text = tokenizer.decode(tokens[:end] if end is not None else tokens, skip_special_tokens=True)
                status, predictions, explanation = parse(text, row['condition'], end is not None)
                if path.exists():
                    save(out / 'attempts' / f'{path.parent.name}_{path.stem}_{datetime.now().timestamp()}.json', json.loads(path.read_text()))
                record = dict(row, metadata=meta, prompt=prompt, input_hash=fp, model=MODEL,
                    model_version=MODEL, revision=REVISION, status=status, response=text,
                    predictions=predictions, explanation=explanation,
                    finish_reason='STOP' if end is not None else 'MAX_TOKENS',
                    generated_tokens=(end + 1) if end is not None else len(tokens),
                    timestamp_utc=datetime.now(timezone.utc).isoformat())
                save(path, record)
                print(f"{row['example_number']} {row['condition']}: {status}", flush=True)
        _, counts = audit()
    return 0 if counts['failed'] == 0 else 2

if __name__ == '__main__':
    sys.exit(main())
