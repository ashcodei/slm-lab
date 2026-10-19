"""Admitted complete published-stream rewards, including post-call checks."""
import json
from pathlib import Path

from .causal_code_reward_v2 import evaluate
from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .public_example_code_reward_v1 import PublicExampleCodeReward


class CausalPublicCodeReward:
    role='train';purpose='execution_reward'

    def __init__(self,root):
        root=Path(root).resolve();folder=root/'reports/causal-code-rewards-v2'
        admission=json.loads((folder/'admission.json').read_text());reader=json.loads((folder/'independent-reader.json').read_text())
        if (admission.get('reward_scoring_admitted') is not True or admission['reward_ledger_sha256']!=file_hash(folder/'observations.jsonl')
                or admission['independent_reader_sha256']!=file_hash(folder/'independent-reader.json')
                or reader.get('all_source_identities_and_raw_control_rewards_reconstructed') is not True):
            raise ValueError('Exact actual counterfactual ledger and separate complete reader required')
        FileBindings(admission['source_files_sha256']).check(full=True)
        original=PublicExampleCodeReward(root);self.candidates=original.candidates;self.references=original.references
        rows=list(map(json.loads,(folder/'observations.jsonl').open()));self.controls={r['source_id']:r['ablation'] for r in rows}
        if len(self.controls)!=len(rows) or set(self.controls)!=set(self.candidates):raise ValueError('Every admitted original function needs an explicit causal disposition')
        self.provenance=dict(original.provenance,reward_admission_sha256=file_hash(folder/'admission.json'),
            counterfactual_ledger_sha256=admission['reward_ledger_sha256'],complete_published_stream_required=True,
            partial_credit_ceiling=.9,genuine_human_preference=False)

    def score(self,source_id,response):
        if source_id not in self.candidates:raise ValueError('Held or unadmitted source cannot supply an optimization reward')
        return evaluate(self.candidates[source_id],self.references[source_id],self.controls[source_id],response)
