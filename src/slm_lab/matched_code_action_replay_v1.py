"""Replay an exact retained action prefix; generate only the missing suffix."""
import copy
import hashlib
import json
import math

from .code_tasks import TASKS


class ActionPrefixReplay:
    def __init__(self,records):
        self.layout=[dict(endpoint=endpoint,frame=frame,case=i+1,language=language,task=task[0])
            for endpoint in ['initial','step512'] for frame in ['completion','instruction']
            for i,(language,task) in enumerate((language,task) for language in ['Python','JavaScript'] for task in TASKS)]
        if not isinstance(records,list) or not 1<=len(records)<=len(self.layout):
            raise ValueError('A nonempty original ordered prefix of the full96-program comparison is required')
        self.records=copy.deepcopy(records)
        for index,row in enumerate(self.records):
            if (any(row.get(key)!=value for key,value in self.layout[index].items())
                    or type(row.get('case')) is not int or not isinstance(row.get('complete_prompt_ids'),list)
                    or not row['complete_prompt_ids'] or row['complete_prompt_ids'][0]!=1
                    or any(type(value) is not int or not 0<=value<32768 for value in row['complete_prompt_ids'])
                    or type(row.get('seconds')) not in (int,float) or not math.isfinite(row['seconds']) or row['seconds']<0):
                raise ValueError('Every retained original full prompt, position, source identity and timing must remain exact')
            self.validate_response(row['actual_response_ids'])
            response=row['actual_response_ids']
            if row.get('stopped_on_EOS') is not (response[-1]==2) or row.get('ended_on_budget') is not (len(response)==512 and response[-1]!=2):
                raise ValueError('The retained original genuine EOS and full action-budget boundary must reconstruct')
        self.delivered=self.retained_uses=self.fresh_uses=0

    @staticmethod
    def validate_response(response):
        if (not isinstance(response,list) or not 1<=len(response)<=512
                or any(type(value) is not int or not 0<=value<32768 for value in response)
                or 2 in response[:-1] or response[-1]!=2 and len(response)!=512):
            raise ValueError('Complete unchanged original response actions and their true EOS/budget boundary required')

    def deliver(self,ids,identity,generate):
        index=self.delivered
        if (index>=len(self.layout) or identity!=self.layout[index] or not callable(generate)
                or ids.ndim!=2 or ids.shape[0]!=1 or str(ids.dtype)!='torch.int64' or ids.device.type!='cpu'):
            raise ValueError('The exact next full original CPU prompt and native generation call are required')
        retained=index<len(self.records)
        if retained:
            row=self.records[index]
            if row['complete_prompt_ids']!=ids[0].tolist():raise ValueError('Retained original prompt IDs changed before replay')
            generated=ids.new_tensor([row['complete_prompt_ids']+row['actual_response_ids']])
            metadata=dict(delivery_origin='retained_original_neural_actions',original_neural_generation_seconds=row['seconds'],
                original_action_record_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                new_neural_generation=False)
        else:
            generated=generate()
            if (generated.ndim!=2 or generated.shape[0]!=1 or generated[0,:ids.shape[1]].tolist()!=ids[0].tolist()):
                raise ValueError('Fresh native generation must retain the entire original prompt')
            self.validate_response(generated[0,ids.shape[1]:].tolist())
            metadata=dict(delivery_origin='new_missing_suffix_neural_actions',new_neural_generation=True)
        self.delivered+=1
        self.retained_uses+=int(retained);self.fresh_uses+=int(not retained)
        return generated,metadata
