#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def normalize_label(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).casefold().strip())
    return text.strip(" \t\n\r.,;:!?\"'")


def deduplicate(labels) -> list[str]:
    out, seen = [], set()
    for value in labels if isinstance(labels, list) else []:
        if not isinstance(value, str):
            continue
        value = value.strip()
        key = normalize_label(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def extract_json_object(text: str):
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S | re.I)
    if fenced:
        stripped = fenced.group(1)
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_response(text: str):
    obj = extract_json_object(text)
    predictions = []
    explanation = ""
    if obj:
        for key in ("types", "entity_types", "predictions"):
            if isinstance(obj.get(key), list):
                predictions = deduplicate(obj[key])
                break
        if isinstance(obj.get("explanation"), str):
            explanation = obj["explanation"].strip()
    if not predictions:
        fallback = []
        for line in text.splitlines():
            m = re.match(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$", line)
            if m:
                fallback.append(m.group(1))
        predictions = deduplicate(fallback)
    return predictions, explanation


def load_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"example_number", "entity_id", "entity_label", "condition", "prompt_path", "metadata_path"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Manifest missing required columns: {sorted(required)}")
    pairs = [(r["entity_id"], r["condition"]) for r in rows]
    if len(set(pairs)) != len(pairs):
        raise SystemExit("Manifest has duplicate entity/condition pairs")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="prompts_v5/all_1k/manifest.csv")
    ap.add_argument("--output", default="results/llama_1k")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=800)
    ap.add_argument("--max-jobs", type=int, default=None, help="Smoke-test only; counts pending jobs")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.batch_size < 1 or args.max_new_tokens < 1:
        ap.error("--batch-size and --max-new-tokens must be positive")

    manifest = (ROOT / args.manifest).resolve()
    out = (ROOT / args.output).resolve()
    rows = load_manifest(manifest)

    jobs = []
    fingerprints = []
    expected = set()
    for row in rows:
        prompt_path = (ROOT / row["prompt_path"]).resolve()
        metadata_path = (ROOT / row["metadata_path"]).resolve()
        prompt = prompt_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("entity_id") != row["entity_id"]:
            raise SystemExit(f"Metadata identity mismatch: {metadata_path}")
        fp = digest(prompt + json.dumps(metadata, sort_keys=True, ensure_ascii=False))
        fingerprints.append(fp)
        path = out / f"example_{int(row['example_number']):04d}" / f"{row['condition']}_result.json"
        expected.add(path)
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if old:
            identity_ok = (
                old.get("input_hash") == fp
                and old.get("entity_id") == row["entity_id"]
                and old.get("condition") == row["condition"]
                and old.get("model") == args.model
                and old.get("model_revision") == args.revision
            )
            if not identity_ok:
                raise SystemExit(f"Existing result does not match this run: {path}")
        jobs.append((row, prompt, metadata, fp, path, old))

    out.mkdir(parents=True, exist_ok=True)
    extras = set(out.glob("example_*/*_result.json")) - expected
    if extras:
        raise SystemExit(f"Unexpected result file in output directory: {next(iter(extras))}")

    config = {
        "model": args.model,
        "revision": args.revision,
        "dtype": "bfloat16",
        "device": "cuda",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "manifest": str(Path(args.manifest)),
        "manifest_sha256": digest(manifest.read_text(encoding="utf-8")),
        "inputs_sha256": digest("".join(fingerprints)),
        "runner_sha256": digest(Path(__file__).read_text(encoding="utf-8")),
    }
    config_path = out / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        # Batch size does not affect the semantic generation configuration, so allow
        # it to change for OOM recovery. Everything else must remain fixed.
        for key, value in config.items():
            if key == "batch_size":
                continue
            if previous.get(key) != value:
                raise SystemExit(f"Run configuration changed at {key}; use a new output directory")
    else:
        atomic_json(config_path, config)

    def completed(job) -> bool:
        old = job[-1]
        return bool(old and old.get("status") == "completed")

    def report() -> tuple[int, int, int]:
        complete = failed = missing = 0
        for row, prompt, metadata, fp, path, old in jobs:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            if data and data.get("status") == "completed":
                complete += 1
            elif data:
                failed += 1
            else:
                missing += 1
        print(f"Completed {complete}/{len(jobs)} | failed {failed} | missing {missing}")
        return complete, failed, missing

    if args.verify_only:
        complete, failed, missing = report()
        return 0 if complete == len(jobs) else 2

    pending = []
    for job in jobs:
        old = job[-1]
        if old and old.get("status") == "completed":
            continue
        if old and not args.retry_failed:
            continue
        pending.append(job)
    if args.max_jobs is not None:
        pending = pending[: args.max_jobs]

    print(f"Manifest jobs: {len(jobs)}")
    print(f"Pending this invocation: {len(pending)}")
    print(f"Output: {out}")
    if not pending:
        report()
        return 0

    lock = out / "RUNNING.lock"
    if lock.exists():
        raise SystemExit(f"Lock exists: {lock}. Remove it only if no runner is active.")
    lock.write_text(str(os.getpid()), encoding="utf-8")

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Run this on the CoMa GPU node.")
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("This CUDA device does not report BF16 support.")

        print("Loading model...")
        print("torch:", torch.__version__)
        print("transformers:", transformers.__version__)
        print("GPU:", torch.cuda.get_device_name(0))
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda")
        model.eval()
        print("Model loaded.")

        model_info = {
            **config,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "loaded_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(out / "model_info.json", model_info)

        start_all = time.time()
        done_this_run = 0
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": job[1]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for job in batch
            ]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True)
            input_width = encoded["input_ids"].shape[1]
            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            encoded = {k: v.to("cuda") for k, v in encoded.items()}

            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise SystemExit(
                    f"CUDA OOM with batch size {args.batch_size}. Rerun the same output directory "
                    "with a smaller --batch-size; completed files will be reused."
                )

            for idx, job in enumerate(batch):
                row, prompt, metadata, fp, path, old = job
                generated_ids = outputs[idx, input_width:].detach().cpu().tolist()
                eos = tokenizer.eos_token_id
                if eos in generated_ids:
                    generated_ids = generated_ids[: generated_ids.index(eos) + 1]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                predictions, explanation = parse_response(response)
                cot_required = "cot" in row["condition"]
                status = "completed" if predictions and (not cot_required or explanation) else "parse_error"
                record = {
                    "example_number": int(row["example_number"]),
                    "entity_id": row["entity_id"],
                    "entity_label": row["entity_label"],
                    "description": row.get("description", ""),
                    "condition": row["condition"],
                    "prompt_path": row["prompt_path"],
                    "metadata_path": row["metadata_path"],
                    "input_hash": fp,
                    "model": args.model,
                    "model_revision": args.revision,
                    "model_version": f"{args.model}@{args.revision}",
                    "prompt": prompt,
                    "metadata": metadata,
                    "predictions": predictions,
                    "status": status,
                    "generation_config": {
                        "do_sample": False,
                        "temperature": 0.0,
                        "max_new_tokens": args.max_new_tokens,
                        "dtype": "bfloat16",
                    },
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "response": response,
                    "raw_response": response,
                    "explanation": explanation,
                    "finish_reason": "EOS" if eos in outputs[idx, input_width:].detach().cpu().tolist() else "MAX_NEW_TOKENS",
                    "usage": {
                        "prompt_tokens": int(prompt_lengths[idx]),
                        "completion_tokens": len(generated_ids),
                        "total_tokens": int(prompt_lengths[idx]) + len(generated_ids),
                    },
                }
                atomic_json(path, record)
                done_this_run += 1
                print(
                    f"[{done_this_run}/{len(pending)}] {row['example_number']} {row['condition']} "
                    f"status={status} types={len(predictions)}"
                )

        elapsed = time.time() - start_all
        print(f"Invocation finished in {elapsed/60:.1f} minutes.")
        complete, failed, missing = report()
        return 0 if failed == 0 and missing == 0 else 2
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
