"""One offline command to audit and reproduce the final study."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from .io import ROOT, prepare_resources, digest, load_records, validate_records

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/derived/final')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    out=args.output.resolve()
    if out == ROOT or (ROOT/'results/external_demo') in (out,*out.parents) or (ROOT/'archive') in (out,*out.parents):
        parser.error('Output must be a separate derived directory, not raw evidence or archive')
    prepare_resources(ROOT)
    subprocess.run([sys.executable,str(ROOT/'validate_external_inputs.py')],cwd=ROOT,check=True)
    if args.validate_only:
        records,_,_=load_records(ROOT)
        print(json.dumps(validate_records(ROOT,records),indent=2))
        return 0
    from .core import recompute
    from .audits import audit_extra
    from .regression import verify
    from .diagnostics import enrich
    from .qualitative import replay
    D,E,S,C,records,attempts,labels,g=recompute(ROOT,out)
    audit_extra(out,records,attempts,labels)
    verify(ROOT,out)
    from .final_comparisons import replay as final_comparisons
    final_comparisons(out)
    print('Computing hierarchy, seed/model gain, person/chain and token diagnostics.',flush=True)
    enrich(out,records,attempts,labels,g)
    replay(out,records)
    from .manuscript_checks import compare
    compare(ROOT,out)
    if not args.no_plots:
        from .figures import render
        render(out)
    code={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src/graph_typing').glob('*.py'))}
    (out/'reproduction_manifest.json').write_text(json.dumps(dict(
        study=json.loads((ROOT/'configs/final_study.json').read_text()),
        raw_manifest_sha256=digest(ROOT/'configs/raw_files.jsonl'),
        resources=json.loads((ROOT/'resources/manifest.json').read_text()),
        python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','networkx','matplotlib']},
        code_sha256=code,inference_calls=0,
        extension_note='Reconstructed final appendix analyses verified against every numeric value in tables A.1-A.6.'),indent=2)+'\n')
    print(f'Final offline replay complete: {out}',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
