"""Continuous-batched vLLM runner. Both experiments share one model load.

Only natural stops with valid JSON count as completed. No RPM throttle.
GPU libraries are imported only after all inputs and resume checks pass.
"""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from run_llama import ROOT, MODEL, REVISION, digest, save, parse

EXPERIMENTS = {
    'main': ('prompts_v6/all_1k/manifest.csv', 'llama_1k_fast'),
    'children': ('prompts_v6/children_removed/manifest.csv', 'llama_children_removed_fast'),
}


def response_schema(condition):
    """Bound generated JSON; retain explanation-first order required by CoT prompts."""
    properties = {}
    if 'cot' in condition:
        properties['explanation'] = {'type': 'string', 'minLength': 1, 'maxLength': 1200}
    properties['types'] = {'type': 'array', 'minItems': 1, 'maxItems': 32,
                           'items': {'type': 'string', 'minLength': 1, 'maxLength': 120}}
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def schema_valid(text, condition):
    try:
        obj = json.loads(text)
        expected = {'types', 'explanation'} if 'cot' in condition else {'types'}
        if not isinstance(obj, dict) or set(obj) != expected:
            return False
        types = obj['types']
        if not isinstance(types, list) or not 1 <= len(types) <= 32:
            return False
        if any(not isinstance(v, str) or not v.strip() or len(v) > 120 for v in types):
            return False
        if 'cot' in condition:
            explanation = obj['explanation']
            if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 1200:
                return False
        return True
    except (ValueError, KeyError, TypeError):
        return False


def budget(prompt_tokens, context, ceiling):
    remaining = context - prompt_tokens
    if remaining < 1:
        raise ValueError('Prompt exceeds configured context; refusing truncation')
    return min(ceiling, remaining) if ceiling else remaining


def result_valid(record, job):
    row, path, prompt, meta, fp = job
    identity = dict(entity_id=row['entity_id'], condition=row['condition'],
                    input_hash=fp, model=MODEL, revision=REVISION, backend='vllm')
    if any(record.get(k) != v for k, v in identity.items()) or record.get('metadata') != meta:
        raise ValueError(f'Result identity mismatch: {path}')
    state, pred, explanation = parse(record.get('response', ''), row['condition'],
                                     record.get('finish_reason') == 'STOP')
    if record.get('generation_config', {}).get('structured_json_schema') is not None:
        if not schema_valid(record.get('response', ''), row['condition']):
            return False
    return (record.get('status') == state == 'completed'
            and pred == record.get('predictions') and explanation == record.get('explanation'))


def load_jobs(manifest, out):
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty manifest')
    jobs, keys, paths = [], set(), set()
    for row in rows:
        key = row['entity_id'], row['condition']
        path = out / f"example_{int(row['example_number']):04d}" / f"{row['condition']}_result.json"
        if key in keys or path in paths:
            raise ValueError('Duplicate manifest pair/output path')
        keys.add(key); paths.add(path)
        prompt = (ROOT / row['prompt_path']).read_text()
        meta = json.loads((ROOT / row['metadata_path']).read_text())
        if meta['entity_id'] != row['entity_id']:
            raise ValueError('Metadata identity mismatch')
        jobs.append((row, path, prompt, meta, digest(prompt + json.dumps(meta, sort_keys=True))))
    if set(out.glob('example_*/*_result.json')) - paths:
        raise ValueError('Unexpected result files outside manifest')
    return jobs


def audit(jobs, expected_seed=None):
    pending, counts = [], dict(completed=0, failed=0, missing=0)
    for job in jobs:
        path = job[1]
        if not path.exists():
            counts['missing'] += 1; pending.append(job)
        elif result_valid(json.loads(path.read_text()), job):
            if expected_seed is not None and json.loads(path.read_text()).get('generation_config', {}).get('seed') != expected_seed:
                raise ValueError(f'Mixed result seeds: {path}')
            counts['completed'] += 1
        else:
            counts['failed'] += 1; pending.append(job)
    return pending, counts


def main(argv=None, engine_factory=None, sampling_factory=None, structured_factory=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=['all', 'main', 'children'], default='all')
    p.add_argument('--output-root', required=True)
    p.add_argument('--seed', type=int, required=True, help='Fixed seed for the entire run; never changed on retries')
    p.add_argument('--chunk-size', type=int, default=128, help='Checkpoint every chunk')
    p.add_argument('--max-num-seqs', type=int, default=32)
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--max-new-tokens', type=int, default=16384,
                   help='0 uses all remaining context; stop tokens still stop early')
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--gpu-memory-utilization', type=float, default=.90)
    p.add_argument('--max-attempts', type=int, default=2, help='Per pending job per invocation')
    p.add_argument('--max-jobs', type=int, help='Smoke test, per experiment')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--structured-json', action='store_true',
                   help='Bounded JSON decoding; use a NEW output root for uniform final runs')
    a = p.parse_args(argv)
    if min(a.chunk_size, a.max_num_seqs, a.max_model_len, a.tensor_parallel_size, a.max_attempts) < 1 or a.max_new_tokens < 0:
        p.error('Invalid positive limits')
    if not 0 < a.gpu_memory_utilization < 1 or (a.max_jobs is not None and a.max_jobs < 1):
        p.error('Invalid memory utilization/max-jobs')
    names = list(EXPERIMENTS) if a.experiment == 'all' else [a.experiment]
    base = dict(model=MODEL, revision=REVISION, backend='vllm', dtype='bfloat16',
                temperature=1.0, seed=a.seed, max_new_tokens=a.max_new_tokens,
                max_model_len=a.max_model_len, tensor_parallel_size=a.tensor_parallel_size,
                max_num_seqs=a.max_num_seqs, chunk_size=a.chunk_size,
                gpu_memory_utilization=a.gpu_memory_utilization,
                max_attempts=a.max_attempts, enable_prefix_caching=True,
                structured_json=a.structured_json,
                schema_sha256=digest(json.dumps([response_schema('standard'), response_schema('cot_graph')], sort_keys=True)) if a.structured_json else None,
                runner_sha256=digest(Path(__file__).read_text()),
                parser_sha256=digest((ROOT / 'scripts/run_llama.py').read_text()))
    base['response_format'] = 'json_object'
    experiments = []
    for name in names:
        mf, dirname = EXPERIMENTS[name]
        manifest, out = ROOT / mf, ROOT / a.output_root / dirname
        jobs = load_jobs(manifest, out)
        config = dict(base, manifest_sha256=digest(manifest.read_text()),
                      inputs_sha256=digest(''.join(j[4] for j in jobs)))
        cp = out / 'run_config.json'
        if cp.exists() and json.loads(cp.read_text()) != config:
            p.error(f'Configuration/inputs changed for {name}; use a new output-root')
        if not cp.exists() and list(out.glob('example_*/*_result.json')):
            p.error('Results exist without run configuration')
        pending, counts = audit(jobs, a.seed)
        print(name, counts, flush=True)
        experiments.append((name, out, jobs, config))
    if a.dry_run:
        return 0
    if a.verify_only:
        return 0 if all(not audit(jobs, a.seed)[0] for _, _, jobs, _ in experiments) else 2
    with ExitStack() as stack:
        for name, out, jobs, config in experiments:
            out.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((out / 'RUNNING.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                p.error(f'Another runner owns {out}')
            cp = out / 'run_config.json'
            if cp.exists() and json.loads(cp.read_text()) != config:
                p.error('Configuration changed while acquiring lock')
        if all(not audit(jobs, a.seed)[0] for _, _, jobs, _ in experiments):
            return 0
        if engine_factory is None:
            from vllm import LLM, SamplingParams
            engine_factory, sampling_factory = LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            structured_factory = StructuredOutputsParams
        if a.structured_json and structured_factory is None:
            raise ValueError('Structured output factory required')
        engine_options = {'structured_outputs_config': {'backend': 'xgrammar',
                           'disable_any_whitespace': True}} if a.structured_json else {}
        engine = engine_factory(model=MODEL, revision=REVISION, tokenizer_revision=REVISION,
            dtype='bfloat16', tensor_parallel_size=a.tensor_parallel_size,
            max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
            gpu_memory_utilization=a.gpu_memory_utilization, enable_prefix_caching=True,
            generation_config='vllm', seed=a.seed, **engine_options)
        tokenizer = engine.get_tokenizer()
        environment = {pkg: importlib.metadata.version(pkg) for pkg in ['vllm', 'torch', 'transformers']} if engine_factory.__module__.startswith('vllm') else {'test_engine': True}
        if not environment.get('test_engine'):
            import torch
            environment.update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
            environment.update(flashinfer_sampler=os.environ.get('VLLM_USE_FLASHINFER_SAMPLER', 'default'),
                               c_compiler=os.environ.get('CC', 'default'))
        for name, out, jobs, config in experiments:
            ep = out / 'environment.json'
            if ep.exists() and json.loads(ep.read_text()) != environment:
                p.error('Runtime changed; use original environment or a new output-root')
            save(out / 'run_config.json', config); save(ep, environment)
        for name, out, jobs, config in experiments:
            pending = audit(jobs, a.seed)[0]
            if a.max_jobs:
                pending = pending[:a.max_jobs]
            started, successes, attempted = time.monotonic(), 0, 0
            for offset in range(0, len(pending), a.chunk_size):
                retry = pending[offset:offset+a.chunk_size]
                for attempt in range(1, a.max_attempts+1):
                    if not retry:
                        break
                    prompts, params = [], []
                    for job in retry:
                        ids = tokenizer.apply_chat_template([{'role':'user', 'content':job[2]}],
                            tokenize=True, add_generation_prompt=True, return_dict=False)
                        ceiling = a.max_new_tokens * (2 ** (attempt-1))
                        structured_options = {'structured_outputs':
                            structured_factory(json_object=True)}
                        params.append(sampling_factory(temperature=1.0, top_p=1, seed=a.seed,
                            max_tokens=budget(len(ids), a.max_model_len, ceiling),
                            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')],
                            **structured_options))
                        prompts.append({'prompt_token_ids': ids})
                    outputs = engine.generate(prompts, params, use_tqdm=True)
                    if len(outputs) != len(retry):
                        raise RuntimeError('Backend returned an incomplete batch')
                    next_retry = []
                    for job, output, param in zip(retry, outputs, params):
                        row, path, prompt, meta, fp = job
                        answer = output.outputs[0]
                        status, pred, explanation = parse(answer.text, row['condition'], answer.finish_reason == 'stop')
                        if status == 'completed' and a.structured_json and not schema_valid(answer.text, row['condition']):
                            status = 'schema_error'
                        if path.exists():
                            save(out / 'attempts' / f'{path.parent.name}_{path.stem}_{time.time_ns()}.json', json.loads(path.read_text()))
                        record = dict(row, metadata=meta, prompt=prompt, input_hash=fp,
                            model=MODEL, model_version=f'{MODEL}@{REVISION} (vLLM)', revision=REVISION,
                            backend='vllm', status=status, response=answer.text,
                            predictions=pred, explanation=explanation,
                            finish_reason='STOP' if answer.finish_reason == 'stop' else str(answer.finish_reason),
                            generated_tokens=len(answer.token_ids), prompt_tokens=len(output.prompt_token_ids),
                            generation_config=dict(temperature=1.0, seed=a.seed, response_format='json_object', max_tokens=param.max_tokens,
                                structured_json_schema=response_schema(row['condition']) if a.structured_json else None),
                            timestamp_utc=datetime.now(timezone.utc).isoformat())
                        save(path, record)
                        attempted += 1; successes += int(status == 'completed')
                        if status != 'completed':
                            next_retry.append(job)
                    retry = next_retry
                minutes = max((time.monotonic()-started)/60, 1e-9)
                rate = successes/minutes
                print(f'{name}: successful={successes} attempts={attempted} completed/min={rate:.1f}', flush=True)
                save(out / 'throughput.json', dict(successful_this_invocation=successes,
                    attempts_this_invocation=attempted, elapsed_minutes=minutes,
                    completed_per_minute=rate, includes_model_load=False))
            _, counts = audit(jobs, a.seed)
            save(out / 'completion.json', dict(expected=len(jobs), **counts))
            print(name, counts, flush=True)
        return 0 if all(not audit(jobs, a.seed)[0] for _, _, jobs, _ in experiments) else 2


if __name__ == '__main__':
    sys.exit(main())
