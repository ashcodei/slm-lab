"""Build a separate complete-response public human alignment curriculum."""
from pathlib import Path
import json
from slm_lab.whole_human_alignment_v2 import prepare

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    result=prepare(ROOT,ROOT/'data/whole-human-alignment-v2')
    print(json.dumps({k:result[k] for k in ('splits','dialogue_source_recovery','original_HH_rows','excluded','training_admitted')},indent=2))
