"""Bounded whole-document candidate storage, not an optimization data loader."""
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

QUOTAS={'code:Python':200_000_000,'code:JavaScript':75_000_000,'code:C++':75_000_000,
        'code:C':50_000_000,'code:Go':25_000_000,'code:Rust':25_000_000,
        'code:Java':25_000_000,'code:TypeScript':25_000_000,'math':250_000_000,'web':250_000_000}


def key(family,language=None):return family+':'+language if family=='code' else family


class CandidateWriter:
    def __init__(self,folder,quotas):
        self.folder=Path(folder)
        if (self.folder.exists() or not quotas or any(not isinstance(k,str) or type(v) is not int or v<1 for k,v in quotas.items())):
            raise ValueError('Fresh candidate storage and explicit positive whole-document training quotas required')
        self.folder.mkdir();self.quotas=dict(quotas);self.train=Counter();self.valid=Counter();self.examples=Counter();self.contexts=Counter()
        self.offsets=dict(train=0,valid=0);self.tokens={r:(self.folder/(r+'.tokens.u16')).open('xb') for r in self.offsets}
        self.records=gzip.open(self.folder/'whole-original-documents.jsonl.gz','xt',encoding='utf-8')
        self.database=sqlite3.connect(self.folder/'exact-original-body-shas.sqlite3')
        self.database.execute('CREATE TABLE documents (sha BLOB PRIMARY KEY) WITHOUT ROWID');self.closed=False

    def complete(self):return all(self.train[k]>=v for k,v in self.quotas.items())

    def full(self,budget_key):return budget_key not in self.quotas or self.train[budget_key]>=self.quotas[budget_key]

    def seen(self,digest):return self.database.execute('SELECT 1 FROM documents WHERE sha=?',(bytes.fromhex(digest),)).fetchone() is not None

    def append(self,entry):
        ids=entry['whole_native_ids'];role=entry['role'];budget_key=key(entry['family'],entry['language'])
        field='code' if entry['family']=='code' else 'text';text=entry['original_publisher_row'][field];digest=hashlib.sha256(text.encode()).hexdigest()
        if (self.closed or budget_key not in self.quotas or role not in self.offsets or entry['optimization_admitted'] is not False
                or entry['human_preferences'] is not False or entry['execution_reward'] is not False
                or entry['original_UTF8_sha256']!=digest or entry['original_public_text']!=text
                or role!=('valid' if int(digest[-8:],16)%100==0 else 'train')
                or not ids or ids[0]!=1 or ids[-1]!=2 or any(type(v) is not int or not 4<=v<32768 for v in ids[1:-1])
                or entry['whole_native_labels']!=[-100,*ids[1:]] or entry['complete_causal_targets']!=len(ids)-1
                or entry['complete_context_tokens']!=len(ids)):
            raise ValueError('Every exact whole unadmitted original source, deterministic role, own native mask and EOS is required')
        if self.full(budget_key):return 'training_quota_already_satisfied'
        if self.seen(digest):return 'exact_original_body_duplicate_within_new_candidate'
        native=np.asarray(ids,dtype='<u2').tobytes();offset=self.offsets[role]
        record={k:v for k,v in entry.items() if k not in ['whole_native_ids','whole_native_labels','original_public_text']}
        record.update(original_text_publisher_field=field,native_file=role+'.tokens.u16',native_token_offset=offset,
            native_token_length=len(ids),native_little_u16_sha256=hashlib.sha256(native).hexdigest(),
            target_mask='Every original token after native BOS, including EOS; BOS is not a target')
        self.tokens[role].write(native);self.records.write(json.dumps(record,ensure_ascii=False,default=str,separators=(',',':'))+'\n')
        self.database.execute('INSERT INTO documents VALUES (?)',(bytes.fromhex(digest),));self.offsets[role]+=len(ids)
        self.examples[(budget_key,role)]+=1;self.contexts[role]+=len(ids)
        (self.train if role=='train' else self.valid)[budget_key]+=len(ids)-1
        return 'whole_original_document_retained_unadmitted'

    def flush(self):
        for stream in self.tokens.values():stream.flush();os.fsync(stream.fileno())
        self.records.flush();os.fsync(self.records.fileno());self.database.commit()

    def summary(self):
        return dict(train_target_tokens_by_quota={k:self.train[k] for k in self.quotas},valid_target_tokens_by_quota={k:self.valid[k] for k in self.quotas},
            actual_train_targets=sum(self.train.values()),actual_valid_targets=sum(self.valid.values()),
            original_documents={k:{r:self.examples[(k,r)] for r in ['train','valid']} for k in self.quotas},
            native_context_token_counts=dict(self.offsets),all_training_quotas_satisfied=self.complete(),
            whole_final_document_may_exceed_quota_no_target_clipping=True,optimization_admitted=False)

    def close(self):
        if not self.closed:
            self.flush();self.records.close()
            for stream in self.tokens.values():stream.close()
            self.database.close();self.closed=True
