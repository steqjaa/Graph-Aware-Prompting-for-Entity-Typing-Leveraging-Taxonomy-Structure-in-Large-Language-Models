"""Restore the full original label inventory from its checksum-pinned gzip."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from graph_typing.io import ROOT,prepare_resources
if __name__=='__main__':
    prepare_resources(ROOT)
    print('Original resource bytes verified.')
