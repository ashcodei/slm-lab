"""Reclaim unused public GPU dependency HTTP download copies only.

Installed packages, local built wheels, datasets and model files are not
eligible. Prepare inventories exact bytes and live-source/open-file guards;
apply revalidates every candidate before deleting paired HTTP cache entries.
This is storage maintenance, not a neural change or quality improvement.
"""
import argparse
from datetime import datetime,timezone
from importlib.metadata import version,PackageNotFoundError
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile

from pip._vendor import msgpack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('/home/jovyan/.cache/pip/http-v2')
OUT=ROOT/'reports/unused-GPU-wheel-cache-v2'
MAIN=ROOT/'reports/public-pretraining-recovery-v1'
RESEARCH=ROOT/'reports/full-function-repair-comparison-CPU-v1'


def normalized(name):return name.lower().replace('_','-').replace('.','-')


def identity(pid):
    tail=Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()
    if tail[0] in {'Z','X'}:raise ValueError('An original work owner is terminal')
    return dict(pid=int(pid),ticks=tail[19])


def owners():
    main=json.loads((MAIN/'plan.json').read_text());research=json.loads((RESEARCH/'plan.json').read_text())
    handoff=json.loads((ROOT/'reports/instance-lineage-recovery-v2/complete.json').read_text())
    original=[main['owner'],research['owner'],*[r['actual_owner'] for r in handoff['actual_ten_new_controllers']]]
    original.extend(r for r in json.loads((MAIN/'children.json').read_text()) if r['subject'].startswith('train-rank-'))
    result=[]
    for before in original:
        actual=identity(before['pid'])
        if str(actual['ticks'])!=str(before['ticks']):raise ValueError('Every original actual process birth must remain live')
        result.append(actual)
    return result


def protected_paths():
    args=['rg','-l','-F','.cache/pip','reports','-g','*.json']
    result=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    if result.returncode not in {0,1}:raise ValueError(result.stderr)
    found=set()
    def collect(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='source_files_sha256' and isinstance(v,dict):found.update(str(Path(n).resolve()) for n in v)
                elif isinstance(v,(dict,list)):collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for name in result.stdout.splitlines():collect(json.loads((ROOT/name).read_text(),parse_int=str))
    return found


def open_cache_files():
    opened=set()
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():continue
        try:
            if process.stat().st_uid!=os.getuid():continue
            for fd in (process/'fd').iterdir():
                try:
                    target=os.readlink(fd)
                    if target.startswith(str(CACHE)):opened.add(target.removesuffix(' (deleted)'))
                except (OSError,FileNotFoundError):pass
        except (PermissionError,FileNotFoundError,ProcessLookupError):pass
    return opened


def stamp(path):
    s=path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or path.is_symlink() or not path.resolve().is_relative_to(CACHE):
        raise ValueError('Only singly linked regular HTTP cache entries are eligible')
    return dict(inode=s.st_ino,bytes=s.st_size,allocated_bytes=s.st_blocks*512,mtime_ns=s.st_mtime_ns)


def observe(body):
    header=body.with_suffix('');before=stamp(body);header_stamp=stamp(header)
    cache_format,data=header.read_bytes().split(b',',1)
    if cache_format!=b'cc=4':raise ValueError('Only the observed paired public HTTP cache format is eligible')
    response=msgpack.loads(data,raw=False)['response'];headers={k.lower():v for k,v in response['headers'].items()}
    project=headers.get('x-pypi-file-project','');release=headers.get('x-pypi-file-version','')
    if response.get('status')!=200 or headers.get('x-pypi-file-package-type')!='bdist_wheel':return None
    if not (normalized(project) in {'torch','triton'} or normalized(project).startswith('nvidia-')):return None
    with zipfile.ZipFile(body) as archive:
        names=[n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names)!=1:return None
        text=archive.read(names[0]).decode('utf-8')
    fields={line.split(': ',1)[0]:line.split(': ',1)[1] for line in text.splitlines() if line.startswith(('Name: ','Version: '))}
    name=fields.get('Name','');installed=version(name)
    if normalized(name)!=normalized(project) or fields.get('Version')!=release or installed==release:return None
    if int(headers.get('content-length','-1'))!=before['bytes']:raise ValueError('Cached public wheel response length differs')
    return dict(body=str(body),header=str(header),body_stamp=before,header_stamp=header_stamp,
        name=name,cached_version=release,installed_version=installed,public_wheel_project=project,
        body_sha256=file_hash(body),header_sha256=file_hash(header),installed_package_bytes_not_targeted=True)


def prepare():
    if OUT.exists() or CACHE.resolve()!=CACHE:raise ValueError('Fresh bounded non-symlink cache plan required')
    before_owners=owners();protected=protected_paths();opened=open_cache_files()
    files=subprocess.check_output(['rg','--files','--hidden','--no-ignore',str(CACHE)],text=True).splitlines()
    selected=[]
    for name in sorted(files):
        body=Path(name)
        if not body.name.endswith('.body') or body.stat().st_size<20*1024**2:continue
        if any(str(p) in protected or str(p) in opened for p in [body,body.with_suffix('')]):continue
        try:row=observe(body)
        except (zipfile.BadZipFile,KeyError,UnicodeError,PackageNotFoundError):continue
        if row is not None:selected.append(row)
    if not selected:raise ValueError('No independently identified unused public GPU wheel copies are eligible')
    OUT.mkdir()
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256={str(Path(__file__)):file_hash(__file__),
        str(MAIN/'plan.json'):file_hash(MAIN/'plan.json'),str(RESEARCH/'plan.json'):file_hash(RESEARCH/'plan.json')},
        actual_live_original_owners=before_owners,cache_root=str(CACHE),selected=selected,
        selected_download_bytes=sum(r['body_stamp']['bytes']+r['header_stamp']['bytes'] for r in selected),
        selected_allocated_bytes=sum(r['body_stamp']['allocated_bytes']+r['header_stamp']['allocated_bytes'] for r in selected),
        live_source_bound_entries_excluded=True,currently_open_cache_entries_excluded=True,
        only_unused_versions_of_already_installed_public_GPU_packages=True,
        installed_files_local_built_wheels_models_and_data_not_eligible=True,goal_achieved=False))
    print(json.dumps(dict(plan=str(OUT/'plan.json'),eligible_files=len(selected),eligible_download_bytes=sum(r['body_stamp']['bytes'] for r in selected))),flush=True)


def apply():
    plan=json.loads((OUT/'plan.json').read_text())
    if (OUT/'complete.json').exists() or (OUT/'removed-entries.jsonl').exists():raise ValueError('Preserve prior applied or partial cache maintenance')
    for path,digest in plan['source_files_sha256'].items():
        if file_hash(path)!=digest:raise ValueError('The exact original safe cache plan implementation/runtime source must remain unchanged')
    if owners()!=plan['actual_live_original_owners']:raise ValueError('Every exact original live owner must remain before cache mutation')
    protected=protected_paths();opened=open_cache_files()
    for row in plan['selected']:
        if any(name in protected or name in opened for name in [row['body'],row['header']]):raise ValueError('An eligible cache copy became source-bound or open')
        if observe(Path(row['body']))!=row:raise ValueError('Every exact original wheel byte, version, inode and metadata must reconstruct before any deletion')
    free_before=shutil.disk_usage(ROOT).free;removed=[]
    with (OUT/'removed-entries.jsonl').open('x') as journal:
        for row in plan['selected']:
            if (stamp(Path(row['body']))!=row['body_stamp'] or stamp(Path(row['header']))!=row['header_stamp']
                    or version(row['name'])!=row['installed_version']):raise ValueError('Original selected cache identities must remain exact at unlink')
            # Remove lookup metadata first, so a subsequent pip request misses
            # this disposable cache entry even if body removal is interrupted.
            Path(row['header']).unlink();Path(row['body']).unlink();removed.append(row)
            journal.write(json.dumps(row)+'\n');journal.flush();os.fsync(journal.fileno())
    after_owners=owners()
    if after_owners!=plan['actual_live_original_owners'] or any(Path(r[k]).exists() for r in removed for k in ['header','body']):
        raise ValueError('Every original owner must remain live and only exact selected entries may disappear')
    for row in removed:
        if version(row['name'])!=row['installed_version']:raise ValueError('Installed package metadata must remain unchanged')
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        removed_cache_entries=removed,removed_download_bytes=plan['selected_download_bytes'],
        removed_allocated_cache_bytes=plan['selected_allocated_bytes'],free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(ROOT).free,
        free_space_delta_can_include_concurrent_training_checkpoint_writes=True,actual_original_live_owners=after_owners,
        installed_package_versions_unchanged=True,models_datasets_and_local_built_wheels_not_targeted=True,
        source_files_sha256={str(OUT/'plan.json'):file_hash(OUT/'plan.json'),str(OUT/'removed-entries.jsonl'):file_hash(OUT/'removed-entries.jsonl')},goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),removed_allocated_cache_bytes=plan['selected_allocated_bytes'],free_bytes_after=shutil.disk_usage(ROOT).free)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','apply']);args=parser.parse_args()
    prepare() if args.mode=='prepare' else apply()
