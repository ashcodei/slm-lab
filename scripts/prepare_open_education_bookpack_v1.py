"""Prepare a separate whole educational-source BPE candidate; no training."""
from pathlib import Path
import json
from slm_lab.open_education_pack_v1 import prepare

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    result=prepare(ROOT,ROOT/'data/open-education-bookpack-v1')
    print(json.dumps({k:result[k] for k in ('splits','roles','unresolved_publisher_attribution_counts','training_admitted')},indent=2))
