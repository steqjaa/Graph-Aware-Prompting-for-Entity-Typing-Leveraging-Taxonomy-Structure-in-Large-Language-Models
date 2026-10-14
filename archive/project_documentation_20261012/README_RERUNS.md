# Thesis experiment reruns

The archive extracts into `thesis/`. It contains the revised prompts, inference runners, evaluation code and required data. All historical results, reports, earlier prompt directories and original project files are retained. New prompts are in `prompts_v6/`; new outputs are written to `results/external_demo/` and do not overwrite historical runs.

## Experiment configuration

The demonstration is Rod Fergusson (`wd:Q18720219`), described in the released WiKC data as "Canadian video game producer". His entity ID is outside the 1,000-entity evaluation cohort. His annotated type chain and selected taxonomy context match the previous demonstration. Source rows and provenance are in `external_example/`.

All 10,353 prompts include this demonstration. Target entities, descriptions, contexts and gold labels are unchanged. The previously zero-shot Harada case now receives the external demonstration.

| Run | Main prompts | Child-removal prompts | Seed |
|---|---:|---:|---|
| Gemini 3.8 Flash | 9,000 | 0 | Omitted; provider-generated |
| Llama 3.1 8B Instruct | 9,000 | 1,353 | 0 |
| Llama 3.1 8B Instruct | 9,000 | 1,353 | 42 |
| Llama 3.1 8B Instruct | 9,000 | 1,353 | 123 |

Temperature is 1.0 throughout. The Llama checkpoint is pinned to revision `0e9e39f249a16976918f6564b8830bc894c89659`. Each run keeps its seed fixed on retries. Total planned accepted responses: 40,059. These runs have not yet been executed.

## Gemini

Extract `thesis.zip` on the Gemini machine, enter `thesis/`, then run:

```bash
bash launch_gemini.sh
```

Python 3 standard library is sufficient for generation. The runner reads `GEMINI_API_KEY` or prompts for it privately in the terminal. Settings: 20 workers, 120 requests/minute, JSON MIME mode, provider-default thinking and no explicit output-token cap.

Results are saved under `results/external_demo/gemini`. After completion, the launcher verifies the results and creates `Gemini_external_demo_results.zip`.

## Llama

Extract `thesis.zip` on CoMa, enter `thesis/` within the GPU allocation, then run:

```bash
bash launch_llama_three_seeds.sh
```

The launcher uses the existing Apptainer container, compiler and Python environment paths from the previous experiment. Seeds 0, 42 and 123 run sequentially. Each seed runs both the main and child-removal conditions. The previous runtime used vLLM 0.31.0, Torch 2.13.0, Transformers 5.17.0 and an H200 NVL; the actual runtime is recorded again.

Results are saved under `results/external_demo/llama_seed_0`, `llama_seed_42` and `llama_seed_123`. After all three seeds complete, the launcher verifies them and creates `Llama_external_demo_three_seeds_results.zip`.

## Resume and verification

Rerunning either launcher resumes saved valid responses. Input hashes and configurations prevent results from different prompts or seeds being mixed. Raw responses, failed attempts and generation metadata are retained.

If a Llama run remains incomplete, inspect its log and saved failure before resuming. Keep its seed fixed; the older fresh-seed recovery script is not part of this protocol.

To create a combined archive, place the Gemini results directory alongside the Llama result directories, then run:

```bash
python3 verify_and_save.py
```

This creates `External_demo_four_runs_results.zip` only after all four runs pass verification. The result archives include raw outputs, attempts, prompts, configuration and provenance.
