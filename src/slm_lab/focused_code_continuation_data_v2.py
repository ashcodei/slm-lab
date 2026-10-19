"""Retain the original replay families and add explicit raw-code/education replay.

Motivated by the measured packed-code NLL drift at the start of the current
instruction phase. This is a new curriculum hypothesis, not evidence that the
drift is fixed. No live or completed v1 training/data artifact is modified.
"""
import hashlib
import json

from .focused_code_continuation_data_v1 import (CODE,EXECUTION,REPLAY as PREVIOUS_REPLAY,
    FocusedCodeContinuationData as PreviousData,FocusedContinuationSchedule as PreviousSchedule,collate_complete)
from .fast_joint_data_v1 import CompleteBucketSchedule

RAW_REPLAY=('packed:original_packed_code','education')
REPLAY=PREVIOUS_REPLAY+RAW_REPLAY


class FocusedCodeContinuationData(PreviousData):
    def __init__(self,root,config,*,purpose='optimization',execution_folder=None):
        super().__init__(root,config,purpose=purpose,execution_folder=execution_folder)
        for name in RAW_REPLAY:
            self.lengths[name]=list(self.replay.lengths[name]);self.targets[name]=list(self.replay.targets[name])
            self.kinds[name]=self.replay.kinds[name]
            if self.kinds[name]!='text' or any(t!=n-1 for n,t in zip(self.lengths[name],self.targets[name])):
                raise ValueError('Complete original raw causal-token replay required')
            self.populations[name]=len(self.lengths[name])


class FocusedContinuationSchedule(PreviousSchedule):
    def __init__(self,data,*,seed=972317,width=16):
        if (type(seed) is not int or type(width) is not int or width<1
                or CODE not in data.lengths or not set(REPLAY)<=data.lengths.keys()):
            raise ValueError('Whole new code and every original plus raw replay family required')
        if data.purpose=='optimization' and EXECUTION not in data.lengths:
            raise ValueError('Production cannot omit the required admitted execution-supervision task')
        def batches(name,salt):
            source=CompleteBucketSchedule({name:list(data.lengths[name])},{name:width},seed+salt,bucket_updates=32)
            return [list(positions) for _,positions in source.entries]
        code=batches(CODE,0);execution=batches(EXECUTION,1) if EXECUTION in data.lengths else []
        replay={};cursors={name:0 for name in REPLAY}
        for i,name in enumerate(REPLAY):
            if not data.lengths[name]:raise ValueError('Every original replay family must remain nonempty')
            replay[name]=batches(name,100+i)
        self.entries=[];used_execution=0
        for i,positions in enumerate(code):
            self.entries.append((CODE,positions));name=REPLAY[i%len(REPLAY)]
            choices=replay[name];selected=list(choices[cursors[name]%len(choices)]);cursors[name]+=1
            self.entries.append((name,selected))
            until=(i+1)*len(execution)//len(code)
            while used_execution<until:
                self.entries.append((EXECUTION,execution[used_execution]));used_execution+=1
        if used_execution!=len(execution):raise ValueError('Every complete auxiliary example must retain its source presentation')
        self.seed=seed;self.width=width;self.phase_for_step=['instruction_and_vision']*len(self.entries)
        self.phase_bounds={'instruction_and_vision':[0,len(self.entries)]}
        self.identity_sha256=hashlib.sha256(json.dumps(self.entries,separators=(',',':')).encode()).hexdigest()
        for family in [CODE]+([EXECUTION] if execution else []):
            actual=[p for name,positions in self.entries if name==family for p in positions]
            if sorted(actual)!=list(range(len(data.lengths[family]))):raise ValueError('No original new-code or execution example may be omitted/repeated')
