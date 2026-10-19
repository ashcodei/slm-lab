"""Future train-only reward catalog; current causal-v2 production is unchanged."""
import json
from pathlib import Path

from .bounded_code_probes_v1 import evaluate
from .causal_code_reward_catalog_v1 import CausalPublicCodeReward
from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash


class BoundedPublicCodeReward:
    role='train';purpose='execution_reward'

    def __init__(self,root):
        root=Path(root).resolve();folder=root/'reports/bounded-code-probes-v1'
        admission=json.loads((folder/'admission.json').read_text());reader=json.loads((folder/'independent-reader.json').read_text())
        if (admission.get('bounded_reference_probe_scoring_admitted') is not True
                or admission['reward_ledger_sha256']!=file_hash(folder/'observations.jsonl')
                or admission['independent_reader_sha256']!=file_hash(folder/'independent-reader.json')
                or reader.get('all_original_reference_outputs_reproduced') is not True or reader.get('all_added_probe_no_ops_failed') is not True):
            raise ValueError('Exact raw probe discovery and separate complete reference/noop readback required')
        FileBindings(admission['source_files_sha256']).check(full=True)
        self.original=CausalPublicCodeReward(root);self.candidates=self.original.candidates;self.references=self.original.references
        rows=list(map(json.loads,(folder/'observations.jsonl').open()));self.rows={r['source_id']:r for r in rows}
        if len(self.rows)!=len(rows) or set(self.rows)!=set(self.candidates):raise ValueError('Every existing original training source must have an explicit probe disposition')
        self.provenance=dict(self.original.provenance,bounded_probe_admission_sha256=file_hash(folder/'admission.json'),
            reference_probe_ledger_sha256=admission['reward_ledger_sha256'],added_machine_reference_agreement_not_publisher_examples=True,
            not_a_formal_domain_or_general_correctness_certificate=True)

    def score(self,source_id,response):
        if source_id not in self.rows:raise ValueError('Only an admitted original train source may supply training probes')
        row=self.rows[source_id]
        if not row['qualified_for_independent_reader']:
            return dict(self.original.score(source_id,response),augmented_probe_status='unavailable_original_published_reward_only',new_probe_count=0)
        return evaluate(self.candidates[source_id],self.references[source_id],self.original.controls[source_id],row['qualification'],response)
