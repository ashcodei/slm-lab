"""Complete new book/code/human text sources for own dense foundation training.

Every packed original train programme/response and every real educational
next-token target is covered in the declared epoch. A global permutation does
not rebalance by repeating rare languages or round the final batch upward.
This is a language-foundation phase; later multimodal/policy integration and
genuine-human alignment remain separate requirements.
"""
import bisect
from collections import Counter
from pathlib import Path

from .admitted_education_v1 import EducationWindows
from .corpus import file_hash
from .own_code_training_stage_v1 import PreparedCodeStream
from .whole_human_data_v2 import WholeHumanData

NAMES=('education','variety','human','program')
REFERENCE_ROUTES=dict(education='general',variety='code',human='general',program='code')


class DenseFoundationSources:
    role='train';purpose='optimization'

    def __init__(self,root,config,education_context=4096):
        self.root=Path(root).resolve();self.config=config
        if type(education_context) is not int or not 2<=education_context<=config.max_text_length:
            raise ValueError('Explicit complete educational window within the actual language context required')
        tokenizer=self.root/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
        self.streams=dict(education=EducationWindows(self.root/'data/open-education-bookpack-v1',file_hash(tokenizer),education_context,'train','optimization'),
            variety=PreparedCodeStream('variety',self.root/'data/codenet-durable-pack-v2',tokenizer,config),
            human=WholeHumanData(self.root/'data/whole-human-alignment-v2','dialogue','train','optimization'),
            program=PreparedCodeStream('generation',self.root/'data/APPS-verified-generation-v1',tokenizer,config))
        self.populations={name:len(stream) for name,stream in self.streams.items()};self.ends=[];total=0
        for name in NAMES:
            stream=self.streams[name]
            if (stream.role,stream.purpose)!=('train','optimization') or not len(stream):
                raise ValueError('Every complete source must have its admitted original optimization role')
            total+=len(stream);self.ends.append(total)
        self.total=total;self.education_context=education_context
        self.variety_languages=[None]*self.populations['variety']
        for language,positions in self.streams['variety'].language_positions.items():
            for p in positions:
                if self.variety_languages[p] is not None:raise ValueError('Each original code position must belong to one language')
                self.variety_languages[p]=language
        if any(v is None for v in self.variety_languages):raise ValueError('All original code-language identities required')

    def __len__(self):return self.total

    def locate(self,position):
        if type(position) is not int or not 0<=position<len(self):raise ValueError('Exact bounded global original source position required')
        i=bisect.bisect_right(self.ends,position);before=0 if i==0 else self.ends[i-1]
        return NAMES[i],position-before

    def information(self,position):
        name,local=self.locate(position);stream=self.streams[name]
        record=dict(global_position=position,family=name,original_train_position=local,reference_route=REFERENCE_ROUTES[name])
        if name=='education':
            start=local*(stream.native_context-1);context=min(stream.native_context,stream.windows.total-start)
            record.update(complete_context_tokens=context,response_EOS_targets=context-1,
                original_first_target_position=start+1,original_last_target_position_exclusive=start+context)
        elif name=='human':
            row=stream.rows[local];native=row['native']['response']
            record.update(complete_context_tokens=native['length'],response_EOS_targets=native['length']-native['target_start'],
                          original_source_id=row['id'],original_language=row['language'])
        else:
            native=stream.pack.index[stream.pack.positions[local]]
            record.update(complete_context_tokens=int(native['length']),response_EOS_targets=int(native['length'])-int(native['target_start']))
            if name=='variety':record['original_language']=self.variety_languages[local]
        return record

    def batch(self,position,device='cpu'):
        name,local=self.locate(position);stream=self.streams[name]
        if name=='education':return stream.batch([local],device)
        if name=='human':return stream.batch([local],'response',device)
        return stream.encoder.batch([dict(packed_code_index=local)],'response',device)

    def budget(self):
        results={}
        for family in NAMES:
            offset=0 if family==NAMES[0] else self.ends[NAMES.index(family)-1]
            rows=[self.information(offset+i) for i in range(self.populations[family])]
            results[family]=dict(original_train_examples=len(rows),complete_context_tokens=sum(r['complete_context_tokens'] for r in rows),
                response_EOS_targets=sum(r['response_EOS_targets'] for r in rows),
                by_original_language=dict(Counter(r['original_language'] for r in rows if 'original_language' in r)))
        return results
