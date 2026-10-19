"""Prepare exact shared model-cache removal; apply needs recorded user approval.

The approved roots contain downloaded external models unused by SLMtraining.
Preserve all own model/source artifacts and all unrelated shared-cache folders.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/shared-pretrained-cache-reclamation-v1'
CACHES = [Path('/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct'),
    Path('/home/jovyan/.cache/huggingface/hub/models--ovedrive--Qwen-Image-Edit-2509-4bit')]


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def inventory(root):
    if root.is_symlink() or root.resolve() != root: raise ValueError('The approved cache root must be an actual directory')
    rows = []
    for path in [root, *sorted(root.rglob('*'))]:
        s = path.lstat()
        if s.st_uid != os.getuid(): raise ValueError('Only the current user cache entries may be reclaimed')
        kind = 'link' if stat.S_ISLNK(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'file' if stat.S_ISREG(s.st_mode) else None
        if kind is None or kind == 'file' and s.st_nlink != 1: raise ValueError('Special and externally hardlinked files are ineligible')
        target = os.readlink(path) if kind == 'link' else None
        if kind == 'link' and not path.resolve().is_relative_to(root): raise ValueError('Only internal snapshot-to-blob cache links are eligible')
        rows.append(dict(path=str(path), kind=kind, device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
            allocated_bytes=s.st_blocks * 512, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, target=target))
    return rows


def protected():
    process = subprocess.run(['rg', '-l', '-F', '.cache/huggingface', 'reports', '-g', '*.json'],
        cwd=ROOT, text=True, capture_output=True)
    if process.returncode not in [0, 1]: raise ValueError(process.stderr)
    found = set()
    def collect(value):
        if isinstance(value, dict):
            for key, v in value.items():
                if key == 'source_files_sha256' and isinstance(v, dict): found.update(str(Path(n).resolve()) for n in v)
                elif isinstance(v, (dict, list)): collect(v)
        elif isinstance(value, list):
            for v in value: collect(v)
    for name in process.stdout.splitlines(): collect(json.loads((ROOT / name).read_text(), parse_int=str))
    return [p for p in sorted(found) if any(Path(p).is_relative_to(root) for root in CACHES)]


def active():
    found = []
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal(): continue
        try:
            if process.stat().st_uid != os.getuid(): continue
            for fd in (process / 'fd').iterdir():
                try:
                    path = os.readlink(fd)
                    if any(path.startswith(str(root) + '/') for root in CACHES): found.append(dict(pid=int(process.name), kind='fd'))
                except OSError: pass
            for line in (process / 'maps').read_text().splitlines():
                if any(str(root) + '/' in line for root in CACHES): found.append(dict(pid=int(process.name), kind='memory_map'))
        except (OSError, UnicodeError): pass
    return found


def owners():
    plan = json.loads((ROOT / 'reports/phase-human-preference-v8/plan.json').read_text())
    children = json.loads((ROOT / 'reports/phase-human-preference-v8/children.json').read_text())
    values = [plan['owner'], *[p for p in children if p['subject'].startswith('train-chosen_CE-rank-')]]
    if len(values) != 5: raise ValueError('This concrete plan expects the original live human owner and all4first-arm ranks')
    result = []
    for before in values:
        fields = Path('/proc', str(before['pid']), 'stat').read_text().split(') ', 1)[1].split()
        if fields[0] in ['Z', 'X'] or fields[19] != str(before['ticks']): raise ValueError('The exact original neural owners changed')
        result.append(dict(pid=before['pid'], ticks=fields[19]))
    return result


def prepare():
    if OUT.exists(): raise ValueError('Preserve every earlier complete or failed shared-cache plan')
    if protected() or active(): raise ValueError('Source-bound or currently used model-cache entries cannot be removed')
    rows = [dict(root=str(root), entries=inventory(root)) for root in CACHES]
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), roots=[str(p) for p in CACHES],
        actual_inventories=rows, actual_original_neural_owners=owners(),
        allocated_regular_file_bytes=sum(p['allocated_bytes'] for r in rows for p in r['entries'] if p['kind'] == 'file'),
        no_SLM_source_bound_paths=True, no_current_user_active_cache_FDs_or_memory_maps=True,
        exact_user_approval_required_before_any_removal=True,
        source_files_sha256={str(Path(__file__)): sha(__file__)}, goal_achieved=False))
    print(str(OUT / 'plan.json'))


def apply():
    plan_path, approval_path = OUT / 'plan.json', OUT / 'user-approval.json'
    plan, approval = json.loads(plan_path.read_text()), json.loads(approval_path.read_text())
    if (OUT / 'complete.json').exists() or (OUT / 'removed-roots.jsonl').exists(): raise ValueError('Preserve previous complete or partial removal')
    if (approval.get('approved') is not True or approval.get('roots') != [str(p) for p in CACHES]
            or approval.get('plan_sha256') != sha(plan_path) or not approval.get('actual_user_reply')):
        raise ValueError('Explicit actual user approval of these exact two shared-cache roots is required')
    if plan['roots'] != [str(p) for p in CACHES] or owners() != plan['actual_original_neural_owners']:
        raise ValueError('The exact approved roots and original live training owners must remain unchanged')
    for name, digest in plan['source_files_sha256'].items():
        if sha(name) != digest: raise ValueError('The original concrete removal implementation changed')
    if protected() or active() or any(inventory(Path(r['root'])) != r['entries'] for r in plan['actual_inventories']):
        raise ValueError('All exact cache bytes/identities must remain unused and unbound before any deletion')
    before = shutil.disk_usage(ROOT).free
    with (OUT / 'removed-roots.jsonl').open('x') as journal:
        for root in CACHES:
            shutil.rmtree(root)
            journal.write(json.dumps(dict(root=str(root), removed=not root.exists())) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
    if owners() != plan['actual_original_neural_owners'] or any(root.exists() for root in CACHES):
        raise ValueError('Original neural owners must remain live and the exact approved roots must be gone')
    atomic(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(),
        removed_roots=plan['roots'], removed_allocated_regular_file_bytes=plan['allocated_regular_file_bytes'],
        free_bytes_before=before, free_bytes_after=shutil.disk_usage(ROOT).free,
        free_space_delta_can_include_concurrent_original_writes=True, original_live_neural_owners_unchanged=True,
        source_files_sha256={str(plan_path): sha(plan_path), str(approval_path): sha(approval_path),
            str(OUT / 'removed-roots.jsonl'): sha(OUT / 'removed-roots.jsonl')}, goal_achieved=False))
    print(str(OUT / 'complete.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'apply'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else apply()
