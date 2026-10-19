"""Consolidate only exact inactive archived copies while retaining every path."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.identical_checkpoint_copy_sharing_v1 import consolidate,metadata

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/inactive-identical-checkpoint-consolidation-CPU-v1'


def protected_paths():
    protected=set();plans=[]
    for name in subprocess.check_output(['rg','--files','reports','-g','plan.json','-g','*-queue.json'],cwd=ROOT,text=True).splitlines():
        try:record=json.loads((ROOT/name).read_text())
        except (ValueError,OSError):continue
        if isinstance(record,dict) and isinstance(record.get('owner'),dict) and not terminal(record['owner']):
            protected.update(str(Path(p).resolve()) for p in record.get('source_files_sha256',{}));plans.append(name)
    return protected,plans


def open_or_mapped(paths):
    paths={str(Path(p).resolve()) for p in paths};observed=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            mapped=(proc/'maps').read_text()
            for path in paths:
                if path in mapped:observed.append(dict(pid=int(proc.name),path=path,kind='existing_mapping'))
            for fd in (proc/'fd').iterdir():
                try:target=os.readlink(fd)
                except FileNotFoundError:continue
                if target in paths:observed.append(dict(pid=int(proc.name),path=target,kind='existing_open_descriptor'))
        except (FileNotFoundError,ProcessLookupError):continue
    return observed


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden archived-copy consolidation required')
    tests=ROOT/'reports/identical-checkpoint-copy-sharing-v1-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('Actual atomic-copy sharing, live-canonical guard and rejection controls must pass')
    groups=defaultdict(list)
    for p in (ROOT/'checkpoints').rglob('*'):
        if p.is_file() and not p.is_symlink() and p.stat().st_size>=1024**3:
            groups[p.stat().st_size].append(p)
    fingerprints=defaultdict(list)
    for size,paths in groups.items():
        if len(paths)<2:continue
        for p in paths:
            with p.open('rb') as stream:
                prefix=stream.read(65536);stream.seek(size//2);middle=stream.read(65536);stream.seek(size-65536);tail=stream.read(65536)
            fingerprints[(size,hashlib.sha256(prefix+middle+tail).hexdigest())].append(p)
    protected,active_plans=protected_paths();candidates=[];skipped=[]
    OUT.mkdir()
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'status.json',dict(owner=me,state='verifying_complete_bytes_of_potential_archived_copies',model_updates=0,goal_achieved=False))
    for (size,fingerprint),paths in fingerprints.items():
        if len(paths)<2:continue
        signatures=defaultdict(list)
        for path in paths:signatures[file_hash(path)].append(path)
        for digest,equal in signatures.items():
            if len(equal)<2:continue
            canonical=next((p for p in equal if str(p.resolve()) in protected),equal[0])
            for destination in equal:
                if destination==canonical:continue
                if (str(destination.resolve()) in protected or canonical.stat().st_ino==destination.stat().st_ino
                        or open_or_mapped([canonical,destination])):
                    skipped.append(dict(source=str(canonical),destination=str(destination),reason='live_bound_or_open_or_already_shared'))
                    continue
                candidates.append(dict(source=str(canonical),destination=str(destination),sha256=digest,bytes=size,
                    original_source_metadata=metadata(canonical),original_destination_metadata=metadata(destination)))
    paths=[Path(__file__),tests,ROOT/'src/slm_lab/identical_checkpoint_copy_sharing_v1.py',ROOT/'tests/test_identical_checkpoint_copy_sharing_v1.py']
    before=shutil.disk_usage(ROOT)
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256={str(p):file_hash(p) for p in paths},
        every_entire_candidate_checkpoint_SHA256_independently_equal=True,actual_complete_candidate_copies=candidates,
        observed_active_plan_paths=active_plans,skipped=skipped,logical_duplicate_candidate_bytes=sum(c['bytes'] for c in candidates),
        filesystem_free_before_bytes=before.free,method='atomic_hard_links_for_inactive_frozen_archives',
        original_block_deduplication_control_not_supported_kernel_status=-95,
        retain_every_scientific_file_path_and_complete_original_SHA256=True,no_model_parameter_or_data_changes=True,goal_achieved=False))
    results=[]
    try:
        with (OUT/'actual-consolidated-copy-records.jsonl').open('x') as journal:
            for candidate in candidates:
                protected,current_plans=protected_paths()
                a,b=Path(candidate['source']),Path(candidate['destination'])
                if open_or_mapped([a,b]):raise ValueError('An archived source/destination gained a live reader; leave all remaining paths untouched')
                result=consolidate(a,b,expected_sha256=candidate['sha256'],live_bound_paths=protected)
                results.append(result);journal.write(json.dumps(result)+'\n');journal.flush()
                atomic_json(OUT/'progress.json',dict(complete_identical_archived_copies=len(results),planned_copies=len(candidates),
                    logical_duplicate_bytes_consolidated=sum(r['logical_duplicate_bytes_consolidated'] for r in results),
                    actual_current_filesystem_free_bytes=shutil.disk_usage(ROOT).free,model_updates=0,goal_achieved=False))
        after=shutil.disk_usage(ROOT)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            actual_identical_archived_copies_consolidated=len(results),actual_results=results,
            logical_duplicate_bytes_consolidated=sum(r['logical_duplicate_bytes_consolidated'] for r in results),
            filesystem_free_before_bytes=before.free,filesystem_free_after_bytes=after.free,
            net_filesystem_free_delta_includes_other_concurrent_writers=after.free-before.free,
            all_original_checkpoint_paths_and_complete_SHA256_values_preserved=True,no_live_bound_destination_inode_replaced=True,
            archived_copies_share_hard_links_not_copy_on_write=True,original_failed_block_sharing_test_log_preserved=True,
            model_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_identical_copies=len(results),
            duplicate_GiB_consolidated=sum(r['logical_duplicate_bytes_consolidated'] for r in results)/1024**3,
            actual_free_GiB=after.free/1024**3,all_original_checkpoint_paths_and_SHA256s_preserved=True)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            completed_byte_identical_consolidations_retained=len(results),no_checkpoint_contents_removed=True,goal_achieved=False));raise


if __name__=='__main__':main()
