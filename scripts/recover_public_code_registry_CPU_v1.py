"""Rebuild the original complete public-code exclusion registry after scratch loss.

Only publisher data is downloaded. Original source/held preparers are reused;
the final registry must match its pre-loss admission SHA exactly. Compact,
lossless question/registry copies and publisher metadata are kept durably.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import cache_public_reasoned_code_train_questions_CPU_v1 as original

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-code-registry-recovery-20261021'
METADATA=original.METADATA
QUESTIONS=original.OUT
HELD=Path('/tmp/slm-public-reasoned-code-held-registry-v1')
EXPECTED='895302b0daffe9a30541ef4212d009b9d49880a83db97883fdf345e22f0bdcea'
GIB=1024**3


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def fetch(url):
    with urllib.request.urlopen(url,timeout=60) as response:return response.read(),response.headers.get('Link','')


def run(script,label):
    with (OUT/(label+'.log')).open('x') as log:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,preexec_fn=lambda:os.nice(19),
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'})
    if result.returncode:raise ValueError('Original '+label+' failed; complete partial evidence is retained')


def retain(path,label):
    target=OUT/label;shutil.copyfile(path,target)
    if sha(path)!=sha(target):raise ValueError('Every durable original receipt byte must match')
    return dict(path=str(target),sha256=sha(target))


def compress(path,label):
    target=OUT/label;before=sha(path)
    with path.open('rb') as source,target.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as output:
        while chunk:=source.read(1024**2):
            output.write(chunk)
            if target.stat().st_size>128*1024**2 or shutil.disk_usage(ROOT).free<8*GIB:
                raise ValueError('Compact durable source allowance or existing writer reserve exceeded')
    digest=hashlib.sha256()
    with gzip.open(target,'rb') as stream:
        while chunk:=stream.read(1024**2):digest.update(chunk)
    if digest.hexdigest()!=before or sha(path)!=before:raise ValueError('Complete lossless durable reconstruction failed')
    return dict(path=str(target),compressed_sha256=sha(target),uncompressed_sha256=before,bytes=target.stat().st_size)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules
            or any(p.exists() for p in [OUT,METADATA,QUESTIONS,HELD])):
        raise ValueError('Fresh CPU-only full-source and held-registry recovery required')
    required=[ROOT/'reports/CodeContests-native-held-text-v2'/name for name in
        ['unchanged-held-text-registry.json','complete.json','additional-component-exclusions.json']]
    required.append(ROOT/'reports/code-contests-complete-curriculum-candidate-v1/all-original-problem-source-dispositions.json')
    if not all(p.is_file() for p in required) or shutil.disk_usage('/tmp').free<64*GIB:
        raise ValueError('All original base exclusions and the complete scratch budget are required')
    OUT.mkdir();METADATA.mkdir();(OUT/'publisher-metadata').mkdir()
    plan=dict(at=datetime.now(timezone.utc).isoformat(),owner_pid=os.getpid(),sources=original.SOURCES,
        expected_original_registry_sha256=EXPECTED,complete_original_train_files_and_held_question_columns_only=True,
        no_source_or_dataset_loader_execution=True,no_GPU_usage=True,model_updates=0,training_admitted=False,
        source_files_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),Path(original.__file__),*required]})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for repo,dataset,revision,_,_ in original.SOURCES:
        slug=repo.replace('/','--')
        info_raw,_=fetch('https://huggingface.co/api/datasets/'+repo+'/revision/'+revision+'?blobs=true')
        info=json.loads(info_raw)
        if info['id']!=repo or info['sha']!=revision or info['lastModified']>='2025-05-01':
            raise ValueError('Exact original pre-release publisher revision required')
        (METADATA/(slug+'-info.json')).write_bytes(info_raw)
        tree=[];url='https://huggingface.co/api/datasets/'+repo+'/tree/'+revision+'?recursive=true&expand=true'
        while url:
            raw,links=fetch(url);tree.extend(json.loads(raw));url=''
            for part in links.split(','):
                if 'rel="next"' in part:url=part.split('<',1)[1].split('>',1)[0]
        (METADATA/(slug+'-tree.json')).write_text(json.dumps(tree,indent=2)+'\n')
        siblings={r['rfilename']:r for r in info['siblings']}
        for name in ['README.md']+([] if dataset=='code_contests' else [repo.rsplit('/',1)[-1]+'.py']):
            raw,_=fetch('https://huggingface.co/datasets/'+repo+'/resolve/'+revision+'/'+name)
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if blob!=siblings[name]['blobId']:raise ValueError('Original publisher text blob must match')
            (METADATA/(slug+'-'+name)).write_bytes(raw)
        print('Pinned publisher metadata recovered:',repo,flush=True)
    for path in METADATA.iterdir():retain(path,'publisher-metadata/'+path.name)
    run('cache_public_reasoned_code_train_questions_CPU_v1.py','train-questions')
    train=json.loads((QUESTIONS/'complete.json').read_text())
    if not train['complete'] or not train['complete_original_train_questions_preserved']:
        raise ValueError('Every original train question must be preserved before held reconstruction')
    run('prepare_public_reasoned_code_held_registry_CPU_v1.py','held-registry')
    complete=json.loads((HELD/'complete.json').read_text())
    registry=HELD/'held-question-registry.json'
    if not complete['complete'] or complete['held_registry_sha256']!=EXPECTED or sha(registry)!=EXPECTED:
        raise ValueError('The complete registry must reproduce the pre-loss admission hash exactly')
    retained={}
    for folder,label,names in [(QUESTIONS,'train',['owner-plan.json','complete.json','original-downloads.json']),
            (HELD,'held',['owner-plan.json','complete.json','original-held-downloads.json'])]:
        for name in names:
            path=folder/name
            if path.exists():retained[label+'-'+name]=retain(path,label+'-'+name)
    copies=dict(whole_train_questions=compress(Path(train['questions_file']),'complete-train-questions.jsonl.gz'),
        whole_held_registry=compress(registry,'held-question-registry.json.gz'))
    receipt=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),original_held_registry_sha256=EXPECTED,
        complete_held_question_stems=complete['total_held_question_stems'],original_registry_bytes_reproduced_exactly=True,
        durable_lossless_copies=copies,retained_original_receipts=retained,all_original_raw_files_still_volatile=True,
        CodeIO_roles_and_whole_target_screen_and_execution_and_admission_still_required=True,
        data_admitted_to_training=False,model_updates=0,goal_achieved=False)
    with (OUT/'complete.json').open('x') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
