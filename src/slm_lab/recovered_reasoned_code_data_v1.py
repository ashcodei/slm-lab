"""Recover an exact previously admitted native corpus after scratch loss.

This does not re-admit sources or pretend lost execution witnesses survived.
Every persisted source/role/token/mask is rebuilt against the original sealed
manifest and a completed persistent policy's source identities. New source
admission still requires the original independent qualification pipeline.
"""
import gzip
import hashlib
import json
from pathlib import Path
import struct

from tokenizers import Tokenizer

from .codenet_checkpoint_v1 import FileBindings
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash
from .public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData,digest,REVISION

MANIFEST='379665024ed1e7cd70c31653ade14070ab7ac10d95ab9630ddaf072353c8bb28'
ADMISSION='8b370b9bcce35831c15c106040a2cc6ae1b9a925dcc8b0a3fcafd1eee8d82db4'


class RecoveredReasonedCodeSFTData(PublicReasonedCodeSFTData):
    recovery_of_existing_admitted_packet=True
    original_raw_witnesses_reexecuted_during_recovery=False

    def __init__(self,root,*,purpose='optimization',views='both'):
        self.root=Path(root).resolve();self.folder=self.root/'data/public-reasoned-code-SFT-v1'
        if purpose not in ['optimization','capacity'] or views!='both':
            raise ValueError('Recovery retains both complete original views and explicit train use')
        manifest_path=self.folder/'manifest.json';admission_path=self.folder/'admission.json'
        if file_hash(manifest_path)!=MANIFEST or file_hash(admission_path)!=ADMISSION:
            raise ValueError('Only the exact pre-loss admitted native packet may be recovered')
        admission=json.loads(admission_path.read_text())
        if (admission.get('admitted_to_training') is not True or admission['admitted_use']!='weak_supervised_coding_only'
                or admission['human_preferences'] is not False or admission['own_model_on_policy'] is not False
                or admission['use_as_RL_reward_oracle_forbidden'] is not True
                or admission['positive_train_rows']!=3675 or admission['separately_reserved_rows']!=425):
            raise ValueError('Existing complete public synthetic SFT-only admission and both original populations required')
        proof=self.root/'checkpoints/public-coding-research-v1/plain_tokens/complete.json'
        published=json.loads(proof.read_text())
        if (not published['complete'] or published['completed_updates']!=230
                or published['sources']['public_manifest_sha256']!=MANIFEST
                or published['sources']['public_admission_sha256']!=ADMISSION):
            raise ValueError('The exact completed persistent policy must bind this original corpus')
        self.pack=CompleteCodePack(self.folder,'train',purpose='training')
        held=CompleteCodePack(self.folder,'validation',purpose='audit')
        with gzip.open(self.folder/'original-rows.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
        if len(rows)!=4100 or len(self.pack)!=3675 or len(held)!=425:
            raise ValueError('All4100original train/reserved views must remain present')
        tokenizer=Tokenizer.from_file(str(self.folder/'tokenizer.json'));groups={};counts={}
        for position,row in enumerate(rows):
            item=self.pack.index[position];role='train' if int(item['role'])==0 else 'validation'
            fixed=row['original_fixed_role'];start=int(item['target_start']);offset=int(item['offset']);length=int(item['length'])
            ids=self.pack.ids[offset:offset+length].astype('int64').tolist()
            source='User: '+row['prompt']+'\nAssistant:'
            head=tokenizer.encode(source,add_special_tokens=False).ids;target=tokenizer.encode(row['response'],add_special_tokens=False).ids
            if (row['token_pack_index']!=position or row['split']!=role or row['source_revision']!=REVISION
                    or row['view'] not in ['code_only','reasoning'] or row['publisher_split']!='train'
                    or fixed['source_role']!=('train' if role=='train' else 'reserved')
                    or fixed['optimization_forbidden'] is not (role=='validation')
                    or fixed['role_fixed_before_fresh_replay'] is not True
                    or row['human_preferences'] is not False or row['own_model_on_policy'] is not False
                    or row['use_as_RL_reward_oracle_forbidden'] is not True
                    or row['external_teacher_weights_loaded_by_this_pipeline'] is not False or row['live_teacher_API_used'] is not False
                    or ids!=[1]+head+target+[2] or start!=1+len(head) or not target
                    or tokenizer.decode(head,skip_special_tokens=False)!=source
                    or tokenizer.decode(target,skip_special_tokens=False)!=row['response']
                    or any(not 4<=value<32768 for value in head+target)
                    or hashlib.sha256(struct.pack('<'+'H'*len(ids),*ids)).hexdigest()!=row['complete_input_ids_sha256']):
                raise ValueError('Every original full text, native token, source role and response/EOS boundary must reconstruct')
            if row['group_id'] in groups and groups[row['group_id']]!=role:raise ValueError('A fixed source component crossed roles')
            groups[row['group_id']]=role;key=role+'_'+row['view']+'_examples';counts[key]=counts.get(key,0)+1
        for key,count in counts.items():
            if admission['actual_native_counts'][key]!=count:raise ValueError('All original view/role counts must agree')
        self.purpose=self._declared_purpose=purpose;self.views=self._declared_views=views
        self._locals=tuple(range(len(self.pack)));self._rows=tuple(rows[int(p)] for p in self.pack.positions)
        self._information=[]
        for position,local in enumerate(self._locals):
            row=self._rows[position];ids,labels=self.pack.sequence(local);start=next(i for i,v in enumerate(labels) if v!=-100)
            native=row['original_native_metadata']
            if (native['complete_context_tokens']!=len(ids) or native['response_EOS_targets']!=len(ids)-start
                    or native['target_start']!=start or labels!=[-100]*start+ids[start:] or labels[-1]!=2):
                raise ValueError('Every complete original train mask and budget must remain exact')
            self._information.append(dict(position=position,source_id=row['id'],view=row['view'],
                original_shard_position=row['original_shard_position'],original_group_id=row['group_id'],
                complete_context_tokens=len(ids),response_EOS_targets=len(ids)-start,original_prompt_positions=start,
                complete_native_input_sha256=row['complete_input_ids_sha256'],source_role='train',
                supervision='public_synthetic_weak_SFT_only',use_as_RL_reward_oracle_forbidden=True))
        runtime={str(self.folder/name):sha for name,sha in self.pack.manifest['files_sha256'].items()}
        for p in [manifest_path,admission_path,proof,Path(__file__).resolve()]:runtime[str(p)]=file_hash(p)
        self.source_files_sha256=runtime;self.guard=FileBindings(runtime)
        self.manifest_sha256=MANIFEST;self.admission_sha256=ADMISSION
        missing=[p for p in admission['source_files_sha256'] if not Path(p).exists()]
        self.recovery_evidence=dict(original_manifest_sha256=MANIFEST,original_admission_sha256=ADMISSION,
            all_original_native_rows_and_texts_rebuilt=4100,complete_train_views=3675,complete_reserved_views=425,
            missing_original_witness_files=len(missing),missing_original_witness_paths=missing,
            no_new_source_or_role_admitted=True,original_raw_source_or_execution_witness_revalidation=False)
        self.provenance_identity=digest(dict(recovery='exact_already_admitted_native_packet_after_scratch_loss',
            persisted_original_sources=admission['source_files_sha256'],recovery_runtime=runtime))
        self._selection_identity=digest(dict(view_policy=views,original_train_local_positions=self._locals))
        self.slots=tuple(range(len(self._locals)))
        budget=self.budget(self.complete_schedule())
        if budget['response_EOS_targets']!=3038462 or budget['complete_context_tokens']!=4797843:
            raise ValueError('Every original source/target budget must remain unchanged after recovery')

    def audit_provenance(self):
        self.guard.check(full=True)
        if file_hash(self.folder/'manifest.json')!=MANIFEST or file_hash(self.folder/'admission.json')!=ADMISSION:
            raise ValueError('The exact existing admitted packet changed after recovery')
