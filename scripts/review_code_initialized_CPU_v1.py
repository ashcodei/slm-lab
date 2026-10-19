"""Read back the actual full v3 prototype weights, Adam and complete sources."""
import json
from pathlib import Path

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.own_aligned_assistant_v3 import FORMAT
from slm_lab.own_code_training_stage_v1 import PreparedCodeStream
from slm_lab.whole_human_data_v2 import WholeHumanData

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/own-aligned-assistant-v3-full-CPU-v1'


def main():
    output=REPORTS/'independent-complete-readback.json'
    if output.exists():raise ValueError('Preserve the previous complete prototype reader')
    torch.set_num_threads(2)
    plan=json.loads((REPORTS/'plan.json').read_text());proof=json.loads((REPORTS/'complete.json').read_text())
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    resume=REPORTS/'resume.pt';trained=REPORTS/'trained-disposable-extensions.pt'
    if file_hash(resume)!=proof['resume_sha256'] or file_hash(trained)!=proof['trained_disposable_extensions_sha256']:
        raise ValueError('Exact actual full-probe checkpoint bytes required')
    saved=torch.load(resume,map_location='cpu',weights_only=True,mmap=True)
    final=torch.load(trained,map_location='cpu',weights_only=True,mmap=True)
    initial=saved['model']['extensions'];parts=final['extensions']
    names=[n for n in initial if n.startswith(('language_linears.','language_norms.'))]
    if (saved['model']['format']!=FORMAT or final['format']!=FORMAT or saved['model']['settings']['language_reference_route']!='original_own_code'
            or final['settings']!=saved['model']['settings'] or saved['steps']!=1 or saved['planned_steps']!=2
            or saved['optimized_names']!=names or len(names)!=385 or set(parts)!=set(initial)
            or final['actual_counts']!=proof['actual_counts'] or saved['model']['actual_counts']!=proof['actual_counts']
            or any(not bool(torch.isfinite(t).all()) or t.shape!=initial[n].shape or t.dtype!=initial[n].dtype for n,t in parts.items())
            or any(not torch.equal(t,initial[n]) for n,t in parts.items() if n not in names)
            or saved['fixed_original_state_sha256']!=proof['fixed_original_digest']):
        raise ValueError('Full actual v3 state, frozen unused tensors, exact optimizer order and original digest must reconcile')
    _validate_adam(saved['optimizer'],[initial[n] for n in names],1);_validate_rng(saved['rng'],'cpu')
    if sum(t.numel() for t in parts.values())!=proof['actual_counts']['extension_unique']:
        raise ValueError('Every distinct new learned value must be counted')
    human=WholeHumanData(ROOT/'data/whole-human-alignment-v2','dialogue','train','optimization')
    code=PreparedCodeStream('generation',ROOT/'data/APPS-verified-generation-v1',ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json',ModelConfig(**final['config']))
    ids,labels=human.sequence(0);program,targets=code.pack.sequence(0)
    expected=dict(human=dict(full_context_tokens=len(ids),response_EOS_targets=sum(v!=-100 for v in labels[1:])),
                  verified_program=dict(full_context_tokens=len(program),response_EOS_targets=sum(v!=-100 for v in targets[1:])))
    if (human.rows[0]['id']!=proof['original_human_ID'] or len(code)!=proof['original_generation_population']
            or saved['sampler_state']!=dict(original_human_position=0,original_verified_program_position=0,completed_updates=1)):
        raise ValueError('Actual original capacity-only source identities and source cursor required')
    for metric in proof['source_metrics']:
        if any(any(metric[name][key]!=value for key,value in wanted.items()) for name,wanted in expected.items()):
            raise ValueError('Every complete original human/program native target normalizer must reconstruct')
    guard.check(full=True)
    atomic_json(output,dict(reader_sha256=file_hash(__file__),actual_counts=proof['actual_counts'],
        retained_resume_steps=1,retained_resume_actual_language_Adam_states=385,
        all215unused_new_tensors_equal_between_saved_update1_and_recovered_final_update2=True,
        every_full_native_human_and_verified_program_BOS_response_EOS_target_count_reconstructed=expected,
        actual_source_bound_GPU_or_CPU_next_update_reported_by_original_probe=True,
        reader_did_not_replay_numerical_forward_or_second_update=True,
        actual_original_probe_complete_sha256=file_hash(REPORTS/'complete.json'),
        checked_resume_sha256=file_hash(resume),checked_disposable_export_sha256=file_hash(trained),
        final_step2_Adam_was_compared_in_original_process_but_not_retained_in_step1_snapshot=True,
        all_capacity_updates_discarded_before_production=True,no_new_held_capability_score=True,candidate_promoted=False,goal_achieved=False))


if __name__=='__main__':main()
