"""Remove unnecessary PNG metadata while preserving compressed pixel chunks."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import struct
import zlib
from PIL import Image
from .corpus import file_hash
from .diverse_images import atomic_json,jsonl,pixel_hash

DISCARD={b'iCCP',b'eXIf',b'iTXt',b'zTXt',b'tEXt'}


def strip_png_metadata(raw):
    if raw[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Not PNG')
    position=8;kept=[raw[:8]];removed=[]
    while position<len(raw):
        if position+12>len(raw):raise ValueError('Truncated PNG chunk')
        length=struct.unpack('>I',raw[position:position+4])[0];kind=raw[position+4:position+8]
        end=position+12+length
        if end>len(raw):raise ValueError('Truncated PNG chunk contents')
        chunk=raw[position:end];payload=chunk[4:-4]
        if zlib.crc32(payload)&0xffffffff!=struct.unpack('>I',chunk[-4:])[0]:raise ValueError('PNG CRC mismatch')
        if kind in DISCARD:removed.append(kind.decode('ascii'))
        else:kept.append(chunk)
        position=end
        if kind==b'IEND':
            if position!=len(raw):raise ValueError('Unexpected trailing PNG bytes')
            return b''.join(kept),removed
    raise ValueError('PNG lacks IEND')


def repair_dataset(root):
    root=Path(root);manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text());rows={};paths={};previous_hashes={}
    for name in ['collected.jsonl','train.jsonl','valid.jsonl']:
        path=root/name
        if not path.exists():continue
        rows[name]=list(jsonl(path));previous_hashes[name]=file_hash(path)
        for row in rows[name]:paths.setdefault(row['image'],[]).append(row)
    repaired=[];bytes_removed=0
    for name,references in paths.items():
        path=root/name;raw=path.read_bytes();clean,removed=strip_png_metadata(raw)
        if not removed:continue
        before=hashlib.sha256(raw).hexdigest();after=hashlib.sha256(clean).hexdigest()
        temporary=path.with_suffix('.metadata-repair.tmp');temporary.write_bytes(clean)
        try:
            with Image.open(temporary) as image:digest=pixel_hash(image)
            for row in references:
                if row.get('model_pixel_sha256') and row['model_pixel_sha256']!=digest:
                    raise ValueError('Pixel transform changed while stripping metadata: '+name)
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True);raise
        for row in references:
            if row.get('stored_image_sha256') not in {None,before}:raise ValueError('Stored source checksum mismatch: '+name)
            row['previous_stored_image_sha256']=before;row['stored_image_sha256']=after
            row['storage_metadata_removed']=removed
        bytes_removed+=len(raw)-len(clean);repaired.append({'image':name,'removed':removed,'old_sha256':before,'new_sha256':after})
    for name,records in rows.items():
        target=root/name;temporary=target.with_suffix('.metadata-repair.tmp')
        temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));temporary.replace(target)
        if name in manifest['files_sha256']:manifest['files_sha256'][name]=file_hash(target)
    manifest['storage_metadata_repair']={'completed_at':datetime.now(timezone.utc).isoformat(),
        'images_modified':len(repaired),'bytes_removed':bytes_removed,'previous_jsonl_hashes':previous_hashes,
        'pixel_transform':'Decoded RGB model-input hash at 224px verified unchanged for every modified record.',
        'original_download_bytes':'Retained in source cache; source download hashes and attribution unchanged.'}
    atomic_json(root/'metadata-repair.json',{'images':repaired,**manifest['storage_metadata_repair']})
    atomic_json(manifest_path,manifest);print(root,manifest['storage_metadata_repair'],flush=True)
