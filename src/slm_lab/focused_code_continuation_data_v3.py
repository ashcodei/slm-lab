"""Add separately admitted source-state CE without replacing any old exposure."""
import hashlib
import json
from pathlib import Path

from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash
from .fast_joint_data_v1 import CompleteBucketSchedule
from .focused_code_continuation_data_v2 import (CODE,EXECUTION,REPLAY,collate_complete,
    FocusedCodeContinuationData as PreviousData,FocusedContinuationSchedule as PreviousSchedule)

STATE='new:public_source_state_prediction'


class FocusedCodeContinuationData(PreviousData):
    def __init__(self,root,config,*,purpose='optimization',execution_folder=None,state_folder=None):
        super().__init__(root,config,purpose=purpose,execution_folder=execution_folder)
        if self.execution is None:
            folder=Path(execution_folder) if execution_folder is not None else self.root/'data/code-contests-public-execution-traces-admitted-v1'
            self.execution=CompleteCodePack(folder,'train','training')
            if self.execution.manifest['provenance'].get('no_private_generated_or_held_inputs_or_outputs_used_as_model_text') is not True:
                raise ValueError('Public-only admitted stdout data required even during full source preparation')
        folder=Path(state_folder) if state_folder is not None else self.root/'data/public-state-traces-admitted-v1'
        self.states=CompleteCodePack(folder,'train','training');admission=json.loads((folder/'admission.json').read_text())
        provenance=self.states.manifest['provenance']
        if (admission.get('admitted_scope')!='Public source-state prediction CE only'
                or admission.get('every_original_source_and_native_target_independently_rebuilt') is not True
                or admission.get('actual_all32953_held_stems_scanned_and_flagged_components_excluded') is not True
                or admission.get('execution_reward_training_admitted') is not False
                or admission.get('human_preference_training_admitted') is not False
                or provenance.get('every_admitted_candidate_has_two_matching_actual_whole_source_state_observations') is not True
                or provenance.get('model_text_from_original_train_program_public_input_and_actual_source_observations_only') is not True):
            raise ValueError('Whole repeated public-state CE with independent source/target/held readback required')
        for name,pack in [(EXECUTION,self.execution),(STATE,self.states)]:
            if file_hash(pack.folder/'tokenizer.json')!=self.code.provenance['tokenizer_sha256']:raise ValueError('One exact own tokenizer required for every source')
            entries=pack.index[pack.positions]
            self.lengths[name]=[int(v) for v in entries['length']]
            self.targets[name]=[int(a)-int(b) for a,b in zip(entries['length'],entries['target_start'])]
            self.kinds[name]='text';self.populations[name]=len(entries)
            if not len(entries) or max(self.lengths[name])>4096 or min(self.targets[name])<2:raise ValueError('Every whole native response and EOS required')
            for filename in ['manifest.json','admission.json']:self.source_files_sha256[str(pack.folder/filename)]=file_hash(pack.folder/filename)
            self.source_files_sha256.update({str(pack.folder/n):h for n,h in pack.manifest['files_sha256'].items()})
        self.source_files_sha256.update(admission['source_files_sha256'])

    def batch(self,name,positions,device='cpu'):
        if name not in {EXECUTION,STATE}:return super().batch(name,positions,device)
        if any(type(p) is not int or not 0<=p<self.populations[name] for p in positions):raise ValueError('Exact admitted original source positions required')
        if not positions:return None
        pack=self.execution if name==EXECUTION else self.states
        result=collate_complete([pack.sequence(p) for p in positions],device)
        if (int(result['attention_mask'].sum())!=sum(self.lengths[name][p] for p in positions)
                or int((result['labels'][:,1:]!=-100).sum())!=sum(self.targets[name][p] for p in positions)):
            raise ValueError('Every actual context, source-state target and EOS must reach the batch')
        return result


class FocusedContinuationSchedule(PreviousSchedule):
    def __init__(self,data,*,seed=972317,width=16):
        if STATE not in data.lengths or EXECUTION not in data.lengths:raise ValueError('Both admitted output and whole-state source tasks required')
        previous=PreviousSchedule(data,seed=seed,width=width)
        state=CompleteBucketSchedule({STATE:list(data.lengths[STATE])},{STATE:width},seed+2,bucket_updates=32)
        state_batches=[list(p) for _,p in state.entries];self.entries=[];seen=0;used=0
        code_updates=sum(name==CODE for name,_ in previous.entries)
        for i,(name,positions) in enumerate(previous.entries):
            self.entries.append((name,list(positions)));seen+=int(name==CODE)
            if i+1==len(previous.entries) or previous.entries[i+1][0]==CODE:
                until=seen*len(state_batches)//code_updates
                while used<until:self.entries.append((STATE,state_batches[used]));used+=1
        if (used!=len(state_batches) or sorted(p for n,ps in self.entries if n==STATE for p in ps)!=list(range(len(data.lengths[STATE])))
                or [(n,p) for n,p in self.entries if n!=STATE]!=previous.entries):
            raise ValueError('Every new state example once and every existing source exposure unchanged required')
        self.seed=seed;self.width=width;self.phase_for_step=['instruction_and_vision']*len(self.entries)
        self.phase_bounds={'instruction_and_vision':[0,len(self.entries)]}
        self.identity_sha256=hashlib.sha256(json.dumps(self.entries,separators=(',',':')).encode()).hexdigest()
