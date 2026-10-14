"""Project paths and historical condition names (no model calls)."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'archive' / 'original'
sys.path.insert(0, str(ORIGINAL))
MAIN_RUN = ORIGINAL / 'generated_ranked_results' / 'meta-llama_Llama-3.1-8B-Instruct__supervisor-v3-local-bf16'
CONDITIONS = ['standard', 'graph_aware_standard', 'cot_no_graph', 'cot_graph', 'contrastive_cot_no_graph', 'contrastive_cot_graph']
LABELS = ['Direct', 'Direct + oracle', 'CoT', 'CoT + oracle', 'Contrastive', 'Contrastive + oracle']
