"""Supply new frozen endpoint paths to the unchanged published-sample driver."""
import json
from pathlib import Path
import sys

import execute_source_generation_controls_v1 as controls
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

MAPPING={'objective_CE':'code-repetition-objective-pilot-v1/CE_control',
         'objective_UL':'code-repetition-objective-pilot-v1/CE_plus_repetition_UL'}


def main():
    if len(sys.argv)!=3 or sys.argv[1]!='--experiment' or sys.argv[2] not in MAPPING:
        raise ValueError('Explicit objective_CE or objective_UL endpoint required')
    key=sys.argv[2]
    # Only report locations are adapted inside this process. Raw model code,
    # source roles, published samples and compiler/runtime/oracle are unchanged.
    before=file_hash(Path(controls.__file__))
    controls.SOURCES.update(MAPPING)
    controls.main()
    assert file_hash(Path(controls.__file__))==before
    folder=controls.ROOT/'reports'/f'{MAPPING[key]}-published-sample-controls'
    atomic_json(folder/'path-adapter-provenance.json',dict(adapter=str(Path(__file__).resolve()),
        adapter_sha256=file_hash(Path(__file__)),original_driver=str(Path(controls.__file__)),
        original_driver_sha256=before,process_local_report_path_mapping=MAPPING,
        original_scoring_or_runtime_source_modified=False,optimizer_updates=0,goal_achieved=False))


if __name__=='__main__':main()
