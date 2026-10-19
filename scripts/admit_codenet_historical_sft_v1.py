"""Admit exact historical source SFT; do not launch a model-training process."""
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder
from slm_lab.codenet_sft_admission_v1 import audit_source_rows,POLICY
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'data/codenet-durable-pack-v2';reports=ROOT/'reports/codenet-historical-sft-admission-v1'
    if reports.exists() or (data/'admission.json').exists():
        raise ValueError('Preserve existing source-admission decisions and partial evidence')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes']
    if shutil.disk_usage(ROOT).free<reserve+32*1024**2:
        raise ValueError('Remaining persistent writer capacity plus32MiB admission output required')
    reports.mkdir()
    published=False
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            new_model_training_started=False,goal_achieved=False,**fields))
    try:
        pack=CompleteCodePack(data,'train',purpose='audit')
        readback_path=ROOT/'reports/codenet-native-contract-pack-v2/independent-readback.json'
        readback=json.loads(readback_path.read_text());manifest_path=data/'manifest.json'
        if (readback['source_files_sha256'][str(manifest_path)]!=file_hash(manifest_path)
                or readback['original_complete_examples_read_back']!=pack.manifest['source_examples']
                or not readback['all_original_programme_bytes_and_full_answer_EOS_tokens_preserved']
                or not readback['all_original_order_roles_groups_and_metadata_preserved']):
            raise ValueError('Exact independent complete source/role readback required')
        qpath=ROOT/'reports/codenet-numba-original-references-v1/source-declared-aot-qualification-v1.json'
        q=json.loads(qpath.read_text())
        native={r['original_archive_member']:r for r in q['source_declared_native_preparation_qualifications']}
        if len(native)!=6:raise ValueError('Six source-declared native contracts required')
        held_folder=ROOT/'data/codenet-algorithm-assessment-v1';held_manifest=held_folder/'manifest.json'
        held=json.loads(held_manifest.read_text());references=held_folder/'references.jsonl'
        if (not held['test_only'] or held['training_or_preference_use']
                or file_hash(references)!=held['files_sha256']['references.jsonl']):
            raise ValueError('Exact original held-only algorithm source identities required')
        paths=[Path(__file__),ROOT/'src/slm_lab/codenet_sft_admission_v1.py',
            ROOT/'src/slm_lab/codenet_complete_context_v1.py',ROOT/'src/slm_lab/codenet_token_pack_v1.py',
            ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'src/slm_lab/training.py',ROOT/'src/slm_lab/model.py',
            ROOT/'src/slm_lab/corpus.py',ROOT/'src/slm_lab/diverse_images.py',
            ROOT/'scripts/prepare_codenet_native_contract_pack_v2.py',ROOT/'src/slm_lab/codenet_native_contracts_v1.py',
            manifest_path,readback_path,qpath,held_manifest,references,
            ROOT/'reports/codenet-durable-pack-v1/reconstruction-check.json',
            ROOT/'data/codenet-durable-pack-v1/manifest.json',
            ROOT/'reports/codenet-gnu-original-references-v1/summary.json',
            ROOT/'reports/codenet-gnu-original-references-v1/independent-population-check.json',
            ROOT/'reports/codenet-numba-original-references-v1/summary.json']
        paths += [ROOT/'tests/test_codenet_sft_admission_v1.py',
                  ROOT/'reports/codenet-historical-sft-admission-v1-focused-tests.log']
        paths+=[data/name for name in pack.manifest['files_sha256']]
        bound={str(p):file_hash(p) for p in paths};guard=FileBindings(bound)
        for name,digest in readback['source_files_sha256'].items():
            if file_hash(name)!=digest:raise ValueError('Original revision/readback source changed')
        token=Tokenizer.from_file(str(data/'tokenizer.json'))
        encoder=CompleteCodeEncoder(token,ModelConfig.load(data/'model-config.json'),ROOT)
        specials={i for i,t in token.get_added_tokens_decoder().items() if t.special}
        atomic_json(reports/'plan.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),
            policy=POLICY,source_files_sha256=bound,
            dataset_manifest_sha256=file_hash(manifest_path),
            admission_scope='Original historically Accepted public-source supervised learning. Source/runtime/oracle qualifications remain explicit; no uniform current execution or correctness certificate.',
            retained_native_execution_qualification='Six source-declared preparation contracts are included; actual preparation has not been executed.',
            held_source_programmes_and_components_excluded_from_training=True,
            occupied_verified_writer=occupied,remaining_writer_reserve_bytes=reserve,
            optimizer_updates=0,new_model_training_started=False,goal_achieved=False))
        status('reencoding every original complete source/mask and checking source-only SFT admission',dataset_admitted_to_training=False)
        def rows():
            with gzip.open(data/'original-rows.jsonl.gz','rt',encoding='utf-8') as f:
                for i,line in enumerate(f):
                    if i%2000==0:
                        guard.check();print('source_admission_rows',i,flush=True)
                    yield json.loads(line)
        facts=audit_source_rows(pack,rows(),encoder,specials,native,
            [json.loads(s) for s in references.open()])
        if facts['training_examples']!=112609 or facts['original_examples']!=123594 or len(facts['language_counts'])!=54:
            raise ValueError('Exact complete54-language source population required')
        guard.check(full=True)
        result=dict(completed_at=datetime.now(timezone.utc).isoformat(),**facts,
            dataset_manifest_sha256=file_hash(manifest_path),source_files_sha256=bound,
            completed_GNU_reference_qualifications=json.loads((ROOT/'reports/codenet-gnu-original-references-v1/summary.json').read_text())['qualification_counts'],
            earlier_corpus_contamination_completely_excluded=False,
            source_or_runtime_qualified_programmes_automatically_labelled_wrong=False,
            synthetic_or_human_preference_labels_created=False,
            current_partial_reference_job_is_additional_quality_research_not_a_completed_certificate=True,
            source_admission_criteria_satisfied=True,admitted_role='train',new_model_training_started=False,
            all_training_stage_baselines_and_replay_storage_lifecycle_requirements_still_apply=True,goal_achieved=False)
        result_path=reports/'source-admission.json';atomic_json(result_path,result)
        guard.check(full=True)
        attestation=dict(admitted_at=datetime.now(timezone.utc).isoformat(),
            dataset_manifest_sha256=result['dataset_manifest_sha256'],admitted_to_training=True,
            admitted_role='train',training_examples=facts['training_examples'],policy=POLICY,
            source_admission_report=str(result_path),source_admission_report_sha256=file_hash(result_path),
            source_quality_qualification='Historical Accepted source targets, not uniformly independently execution-verified demonstrations or preference labels.',
            actual_native_preparation_executed=False,
            earlier_corpus_contamination_completely_excluded=False,
            optimizer_updates=0,new_model_training_started=False,goal_achieved=False)
        atomic_json(data/'admission.json',attestation)
        published=True
        admitted=CompleteCodePack(data,'train',purpose='training')
        if (len(admitted)!=112609
                or any(int(admitted.index[p]['role'])!=0 for p in admitted.positions)):
            raise ValueError('Admitted fixed training view differs')
        for role in ['validation','test']:
            try:CompleteCodePack(data,role,purpose='training')
            except ValueError:pass
            else:raise ValueError('Held source role was exposed for optimization')
        atomic_json(reports/'reader-view-check.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
            dataset_manifest_sha256=file_hash(manifest_path),admission_sha256=file_hash(data/'admission.json'),
            actual_training_view_examples=len(admitted),every_returned_source_role_is_train=True,
            validation_and_test_optimization_requests_rejected=True,optimizer_updates=0,
            new_model_training_started=False,goal_achieved=False))
        status('historical-source SFT admitted; model training remains unstarted and separately gated',
            dataset_admitted_to_training=True,training_examples=len(admitted),policy=POLICY,
            admission_sha256=file_hash(data/'admission.json'))
        print('historical_source_SFT_admitted',len(admitted),'languages',len(facts['language_counts']),flush=True)
    except Exception as error:
        if published:
            revoked=json.loads((data/'admission.json').read_text())
            revoked.update(admitted_to_training=False,revoked_at=datetime.now(timezone.utc).isoformat(),
                           revocation_reason='Post-publication reader verification failed: '+str(error))
            atomic_json(data/'admission.json',revoked)
        status('source SFT admission stopped; inspect preserved evidence',
               dataset_admitted_to_training=False,error=str(error));raise


if __name__=='__main__':main()
