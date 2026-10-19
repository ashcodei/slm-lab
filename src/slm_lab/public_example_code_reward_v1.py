"""Own-output code rewards on admitted published training examples only.

The original implementation stays in the parent. Only generated code, declared
stdlib context and original example commands enter the confined worker. Setup
statements earn no reward: only examples known to exercise the original entry
point contribute. Finite example rewards do not prove unseen correctness.
"""
import copy
import gzip
import json
from pathlib import Path

from .code_quality import instruction_code
from .corpus import file_hash
from .documented_doctest_v1 import execute,sha


def evaluate_candidate(candidate,reference_observation,response,*,seconds=5):
    if candidate['source'].get('original_partition')!='train' or candidate['source'].get('language')!='Python':
        raise ValueError('Held examples can never supply a code-RL training reward')
    if (reference_observation.get('source_id')!=candidate['source']['id']
            or reference_observation.get('program_sha256')!=candidate['program_sha256']
            or not reference_observation.get('all_original_examples_passed')):
        raise ValueError('Previously verified original source/example identity required')
    scored=[i for i,r in enumerate(reference_observation['raw_results']) if r['entry_body_calls']>0]
    if not scored:raise ValueError('Original published examples must actually exercise their entry point')
    if not isinstance(response,str):raise ValueError('Actual decoded own response required')
    code=instruction_code(response)
    result=dict(format='own-published-train-example-code-reward-v1',original_source_id=candidate['source']['id'],
        original_retained_program_sha256=candidate['program_sha256'],own_response_sha256=sha(response),own_extracted_code_sha256=sha(code),
        scored_original_example_indices=scored,scored_original_examples=len(scored),reward=0.,
        passed_scored_original_examples=0,expected_outputs_stay_outside_worker=True,
        original_reference_implementation_not_sent_to_worker=True,published_train_examples_only=True,
        complete_original_example_setup_preserved=True,setup_statements_earn_no_reward=True,
        human_preference_label=False,external_model_teacher_used=False,
        supported_contract='One complete Python entry function, nested helpers and the declared stdlib context.',
        finite_examples_not_general_correctness_or_efficiency=True,goal_achieved=False)
    # A derived runtime record binds its GENERATED body; it is never written
    # over the retained source or represented as an original human target.
    generated=copy.deepcopy(candidate);source=dict(candidate['source']);source.update(id='own-response:'+source['id'],
        response='```python\n'+code+'\n```',annotation_origin='Own policy output; not an original source implementation.')
    generated.update(source=source,program=code,program_sha256=sha(code),existing_formatted_source_target_sha256=sha(source['response']))
    try:observation=execute(generated,seconds)
    except (ValueError,SyntaxError) as e:
        return dict(result,status='generated_function_contract_or_context_failure',qualification=str(e))
    raw=observation.get('raw_results',[]);flags=observation.get('example_passes',[])
    if len(raw)==candidate['whole_original_example_count'] and len(flags)==len(raw):
        passed=sum(bool(flags[i]) and raw[i]['entry_body_calls']>0 for i in scored)
        result.update(reward=passed/len(scored),passed_scored_original_examples=passed)
    return dict(result,status=observation['status'],raw_generated_execution=observation,
        all_complete_original_examples_passed=observation['all_original_examples_passed'])


class PublicExampleCodeReward:
    role='train';purpose='execution_reward'
    def __init__(self,root):
        root=Path(root).resolve();folder=root/'data/documented-doctest-verified-v1'
        admission=folder/'admission.json';reader=root/'reports/documented-doctest-verified-v1/independent-complete-source-and-runtime-readback-v1.json'
        r=json.loads(reader.read_text());a=json.loads(admission.read_text())
        if (a.get('admitted_to_training') is not True or not r.get('all_reexecuted_examples_passed')
                or r['admitted_pack_sha256']!=file_hash(admission) or r['manifest_sha256']!=file_hash(folder/'manifest.json')):
            raise ValueError('Exact admitted functions and complete independent original-example replay required')
        with gzip.open(folder/'original-rows.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
        admitted={row['parent_source']['id']:row for row in rows}
        candidates=list(map(json.loads,(root/'data/documented-doctest-stage-v1/candidates.jsonl').open()))
        self.candidates={c['source']['id']:c for c in candidates if c['source']['id'] in admitted}
        original=list(map(json.loads,(root/'reports/documented-doctest-verified-v1/observations.jsonl').open()))
        self.references={o['source_id']:o for o in original if o['source_id'] in admitted}
        if len(self.candidates)!=a['complete_training_programs'] or set(self.candidates)!=set(admitted) or set(self.references)!=set(admitted):
            raise ValueError('Every admitted original train function needs its actual source and original raw execution')
        for key,c in self.candidates.items():
            if admitted[key]['parent_source']!=c['source'] or c['source']['original_partition']!='train':raise ValueError('Original source/role changed')
        self.provenance=dict(admission_sha256=file_hash(admission),independent_runtime_reader_sha256=file_hash(reader),
            own_output_reward_not_fresh_human_preference=True,benchmark_answers_never_supply_training_rewards=True)

    def score(self,source_id,response):
        if source_id not in self.candidates:raise ValueError('Only an admitted original training function may receive a training reward')
        return evaluate_candidate(self.candidates[source_id],self.references[source_id],response)
