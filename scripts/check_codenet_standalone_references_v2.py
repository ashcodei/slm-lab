"""Preserve v1 and repeat the identical source selection after a runner correction."""
from pathlib import Path

from check_codenet_standalone_references_v1 import ROOT, main


if __name__ == '__main__':
    main(folder_name='codenet-standalone-reference-pilot-v2', extra_bound_paths=(Path(__file__),),
         previous_selection_path=ROOT / 'reports/codenet-standalone-reference-pilot-v1/selection.jsonl')
