import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_llama_fast import load_jobs, audit, MODEL, REVISION

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gemini', default='results/gemini_1k_fresh')
    p.add_argument('--llama', default='results/llama_1k_fast')
    p.add_argument('--children', default='results/llama_children_removed_fast')
    p.add_argument('--gemini-only', action='store_true')
    a = p.parse_args()
    run('scripts/score_gemini.py', '--results', a.gemini)
    run('plot_stories.py', '--results', a.gemini, '--primary-label', 'Gemini 3.8 Flash')
    if a.gemini_only:
        return
    configurations = []
    for manifest, results in [('prompts_v5/all_1k/manifest.csv', a.llama),
                              ('prompts_v5/children_removed/manifest.csv', a.children)]:
        root = ROOT / results
        jobs = load_jobs(ROOT / manifest, root)
        pending, counts = audit(jobs)
        if pending:
            raise SystemExit(f'Final Llama results are incomplete: {results}: {counts}')
        config = json.loads((root / 'run_config.json').read_text())
        if (config.get('model'), config.get('revision'), config.get('backend')) != (MODEL, REVISION, 'vllm'):
            raise SystemExit('Unexpected final Llama model/backend')
        configurations.append({k:v for k,v in config.items() if k not in ['manifest_sha256','inputs_sha256']})
        run('scripts/score_gemini.py', '--manifest', manifest, '--results', results)
    if configurations[0] != configurations[1]:
        raise SystemExit('Child-removal and main Llama configurations differ')
    run('plot_stories.py', '--results', a.llama, '--primary-label', 'Llama 3.1 8B Instruct (vLLM)',
        '--compare-results', a.gemini, '--compare-label', 'Gemini 3.8 Flash',
        '--children-results', a.children)
    run('scripts/comparison_report.py', '--gemini', a.gemini, '--llama', a.llama,
        '--children', a.children)
    run('scripts/graph_context_report.py', '--gemini', a.gemini, '--llama', a.llama)
    run('scripts/entity_distribution.py')
    run('scripts/plot_required_metrics.py', '--gemini', a.gemini, '--llama', a.llama,
        '--children', a.children)


if __name__ == '__main__':
    main()
