"""Actual original train-only source delivery for the paced joint recipe.

Reuse the independently admitted encoders/pixels/native packs unchanged.
Augment their delivery with every complete corrective negative; no models,
optimizers or GPU jobs are constructed or admitted by this source class.
"""
from dataclasses import asdict
from pathlib import Path
import json

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .corrected_multimodal_batches_v1 import CorrectedMultimodalBatches
from .dense_human_preference_v1 import pair_information
from .exact_waiting_process_retirement_v1 import terminal
from .joint_visual_source_delivery_v1 import OriginalVisualSources
from .mixed_coding_SFT_data_v1 import MixedCodingSFTData
from .complete_joint_source_schedule_v1 import digest
from .spread_joint_source_schedule_v1 import FORMAT
from .whole_human_data_v2 import WholeHumanData


def read(path):return json.loads(Path(path).read_text())


class CompleteJointTrainingDataV1:
    role='train'
    purpose='optimization'

    def __init__(self,root,config,*,word_expansion):
        self.root=Path(root).resolve();self.config=config
        if (config.vocab_size!=32768 or config.max_text_length!=4096 or config.image_size!=224
                or config.max_image_size!=448 or config.patch_size!=16 or word_expansion!=5):
            raise ValueError('All original native vocabulary/context,448px/64x448px and140frame source protocols are required')
        self.folder=self.root/'reports/spread-joint-source-schedule-CPU-v1'
        self.original=self.root/'reports/complete-joint-source-schedule-CPU-v1'
        self.closed=read(self.folder/'complete.json');owner=read(self.folder/'plan.json')['owner']
        if (not terminal(owner) or (self.folder/'failure.json').exists() or self.closed.get('complete') is not True
                or self.closed.get('production_training_admitted') is not False
                or file_hash(self.folder/'joint.json')!=self.closed['joint_sha256']):
            raise ValueError('The exact complete paced source recipe and its independent predecessor reader are required')
        self.files=dict(self.closed['source_files_sha256'])
        for path in [self.folder/'complete.json',self.folder/'plan.json',self.folder/'joint.json']:
            self.files[str(path)]=file_hash(path)
        self.guard=FileBindings(self.files)
        self.joint=read(self.folder/'joint.json')
        if self.joint['format']!=FORMAT or self.joint['actual_joint_updates']!=5362 or len(self.joint['entries'])!=5362:
            raise ValueError('The full5362step original joint horizon must remain')
        self.streams={name:read(self.original/(name+'.json')) for name in ['code','visual','human']}
        for name,part in self.streams.items():
            if digest(part)!=self.joint['component_streams_sha256'][name]:raise ValueError('The entire original component stream differs')
        self.code=MixedCodingSFTData(self.root)
        self.visual_adapter=CorrectedMultimodalBatches(self.root,config,purpose='optimization')
        self.visual=OriginalVisualSources(self.visual_adapter)
        self.visual_entries=read(self.root/'reports/full-multimodal-corrected-source-index-CPU-v1/complete-corrected-single-epoch-schedule.json')['entries']
        self.human=WholeHumanData(self.root/'data/whole-human-alignment-v2','preferences','train','optimization')
        tokenizers=[self.root/'data/tokenizer-corpus/tokenizer.json',self.code.weak.folder/'tokenizer.json',self.code.human.tokenizer_path,self.code.reasoning.folder/'tokenizer.json']
        self.tokenizer_sha256=file_hash(tokenizers[0])
        if len({file_hash(p) for p in tokenizers})!=1 or self.human.manifest['tokenizer_sha256']!=self.tokenizer_sha256:
            raise ValueError('Every actual joint source must use the exact same own native vocabulary')
        self.identity_sha256=digest(dict(paced_joint_sha256=file_hash(self.folder/'joint.json'),source_files=self.files,
            configuration=asdict(config),word_expansion=word_expansion,tokenizer_sha256=self.tokenizer_sha256))
        self.guard.check(full=True)

    def __len__(self):return 5362

    def metadata(self,step):
        if type(step) is not int or not 0<=step<len(self):raise ValueError('An original complete joint source step is required')
        row=self.joint['entries'][step];result={}
        for reference in row['components']:
            name=reference['stream'];entry=self.streams[name]['entries'][reference['source_step']-1]
            if digest(entry)!=reference['original_entry_sha256']:raise ValueError('An original full component entry changed')
            if name=='code':
                actual=self.code.metadata(entry['original_positions'],'balanced_sources')
                if (actual['complete_context_tokens']!=entry['complete_context_tokens'] or actual['response_EOS_targets']!=entry['response_EOS_targets']
                        or actual['actual_global_weighted_targets']!=entry['original_global_positive_weighted_targets']
                        or digest(actual['complete_sources'])!=entry['original_complete_source_information_sha256']):
                    raise ValueError('Every complete original code source and weighted target denominator must match')
            elif name=='human':
                info=[pair_information(self.human,p) for p in entry['original_positions']]
                if (digest(info)!=entry['complete_original_pair_information_sha256']
                        or sum(r['chosen_response_EOS_targets'] for r in info)!=entry['chosen_response_EOS_targets']
                        or sum(r['rejected_response_EOS_targets'] for r in info)!=entry['rejected_response_EOS_targets']):
                    raise ValueError('Every whole original human pair and both complete response budgets must match')
            else:
                original=self.visual_entries[reference['source_step']-1]
                if (digest(original)!=entry['original_corrected_entry_sha256'] or original['original_local_positions']!=entry['original_positions']):
                    raise ValueError('Every original complete visual batch must retain its source positions and index identity')
            result[name]=dict(reference=reference,entry=entry)
        return dict(step=step+1,source_entry_sha256=digest(row),components=result)

    def prepare_visual(self,component,*,rank,world,device):
        reference,declared=component['reference'],component['entry']
        entry=self.visual_entries[reference['source_step']-1]
        payload,original=self.visual.prepare(entry,rank=rank,world=world,device=device)
        positions=original['global_positions'];family=entry['family'];negative_lengths=[];negative_targets=[];corrections=[]
        indexed=self.visual_adapter.index[self.visual_adapter.positions[family][positions]]
        for value in indexed['correction_index']:
            if int(value)<0:corrections.append(None);negative_lengths.append(0);negative_targets.append(0)
            else:
                negative=self.visual_adapter.corrections[int(value)]['complete_original_negative_native']
                corrections.append(negative);negative_lengths.append(len(negative['input_ids']));negative_targets.append(negative['response_EOS_targets'])
        source=dict(source_role='train',purpose='optimization',source_sha256=reference['original_entry_sha256'],positions=positions,
            context_lengths=original['global_context_lengths'],response_EOS_targets=original['global_response_EOS_targets'],image_size=448,
            negative_context_lengths=negative_lengths,negative_response_EOS_targets=negative_targets)
        if sum(negative_targets)!=declared['complete_original_negative_response_EOS_targets']:
            raise ValueError('Every original corrective negative EOS target must reach actual delivery')
        negatives={}
        for slot,negative in enumerate(corrections[rank::world]):
            if negative is not None:
                negatives[slot]=dict(input_ids=torch.tensor([negative['input_ids']],dtype=torch.long,device=device),
                    labels=torch.tensor([negative['labels']],dtype=torch.long,device=device),
                    attention_mask=torch.ones(1,len(negative['input_ids']),dtype=torch.bool,device=device),pixel_values=payload['native']['pixel_values'][slot:slot+1])
        if family=='words':
            source.update(word_height=64,word_width=448,UTF8_bytes=original['global_word_UTF8_bytes'],
                UTF8_sha256=original['global_word_UTF8_sha256'],CTC_frames=140)
            word={k:payload['word'][k] for k in ['word_pixels','texts']} if payload else None
            return dict(kind='word',native=payload['native'] if payload else None,word=word,matching=None,negatives=negatives,source=source)
        if family=='images:additional_photo_captions':
            source.update(pixel_keys=original['pixel_keys'],captions=original['captions'])
            matching={k:payload['matching'][k] for k in ['text_ids','text_mask']} if payload else None
            return dict(kind='photo',native=payload['native'] if payload else None,word=None,matching=matching,negatives=negatives,source=source)
        return dict(kind='native',native=payload['native'] if payload else None,word=None,matching=None,negatives=negatives,source=source)
