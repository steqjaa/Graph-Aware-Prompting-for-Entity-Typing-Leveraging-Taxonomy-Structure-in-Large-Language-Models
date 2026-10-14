"""Numerical replay gates against immutable supplied revision tables."""
import json
import numpy as np
import pandas as pd

TABLES={
 'condition_summary.csv':['run','condition'],
 'comparisons.csv':['kind','run','style','stratum'],
 'seed_summary.csv':['condition'],
 'entities.csv':['example'],
 'failure_sensitivity.csv':['run','style'],
}

def verify(root,out):
    source=root/'archive/recovered_analysis/revision_20261013/analysis'
    checks=[]
    for name,keys in TABLES.items():
        expected=pd.read_csv(source/name);actual=pd.read_csv(out/name)
        common=[k for k in keys if k in expected.columns]
        if set(expected.columns)!=set(actual.columns):raise ValueError(f'Column regression: {name}')
        a=expected.fillna({k:'' for k in common}).sort_values(common).reset_index(drop=True)
        b=actual.fillna({k:'' for k in common}).sort_values(common).reset_index(drop=True)
        if len(a)!=len(b):raise ValueError(f'Row-count regression: {name}')
        for c in a:
            if pd.api.types.is_numeric_dtype(a[c]):
                if not np.allclose(a[c],b[c],rtol=0,atol=1e-12,equal_nan=True):
                    delta=np.nanmax(np.abs(a[c].astype(float)-b[c].astype(float)))
                    raise ValueError(f'Numerical regression: {name}:{c}; max difference {delta}')
            elif not a[c].fillna('').equals(b[c].fillna('')):
                raise ValueError(f'Value regression: {name}:{c}')
        checks.append(dict(table=name,rows=len(a),columns=len(a.columns),status='matched',absolute_tolerance=1e-12))
    (out/'regression_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('All five recovered reference tables match within 1e-12.',flush=True)
    return checks
