import hashlib
import json
import secrets
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import run_llama_json_mode as runner
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

retry_seed = secrets.randbelow(2**31 - 1) + 1
seeds = defaultdict(deque)
original_save = runner.save

def sampling(**kwargs):
    kwargs["seed"] = retry_seed
    return SamplingParams(**kwargs)

class Engine:
    def __init__(self, **kwargs):
        self.engine = LLM(**kwargs)

    def get_tokenizer(self):
        return self.engine.get_tokenizer()

    def generate(self, prompts, params, **kwargs):
        outputs = self.engine.generate(prompts, params, **kwargs)
        for output, param in zip(outputs, params):
            seeds[output.outputs[0].text].append(param.seed)
        return outputs

Engine.__module__ = "vllm"

def save(path, obj):
    if (
        path.name.endswith("_result.json")
        and "attempts" not in path.parts
        and "generation_config" in obj
    ):
        queue = seeds[obj["response"]]
        assert queue, "Missing seed provenance; refusing to save"
        obj["generation_config"]["seed"] = queue.popleft()
        obj["retry_provenance"] = {
            "reason": "Fresh-seed retry of unsuccessful response",
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    original_save(path, obj)

runner.save = save
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
original_save(
    runner.ROOT / "results/json_mode_final" / f"retry_seed_{stamp}.json",
    {"seed": retry_seed, "scope": "Unsuccessful jobs only",
     "settings": "Original settings; sampling seed overridden"}
)
print("Fresh retry seed:", retry_seed, flush=True)
raise SystemExit(runner.main(
    ["--max-new-tokens", "0",
     "--output-root", "results/json_mode_final"],
    engine_factory=Engine,
    sampling_factory=sampling,
    structured_factory=StructuredOutputsParams,
))
