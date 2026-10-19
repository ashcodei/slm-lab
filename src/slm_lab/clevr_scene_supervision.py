"""Dense native descriptions computed only from original rendered-scene metadata."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from tokenizers import Tokenizer
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .model import ModelConfig
from .training import Encoder

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ('Describe every object in the image from left to right by its projected center. '
          'Give each object\'s size, color, material, and shape. Finish with counts of cubes, cylinders, and spheres.')
ATTRIBUTES = {'size': {'small','large'}, 'color': {'gray','red','blue','green','brown','purple','cyan','yellow'},
              'material': {'rubber','metal'}, 'shape': {'cube','cylinder','sphere'}}


def description(scene):
    objects = scene['objects']
    if not 3 <= len(objects) <= 10: raise ValueError('Original scene object count outside defined CLEVR range')
    for obj in objects:
        if any(obj[key] not in values for key,values in ATTRIBUTES.items()): raise ValueError('Unknown original scene attribute')
        coordinates = obj['pixel_coords']
        if (len(coordinates) != 3 or not all(math.isfinite(float(x)) for x in coordinates)
                or not 0 <= coordinates[0] < 480 or not 0 <= coordinates[1] < 320):
            raise ValueError('Original projected object center outside source image')
    ordered = sorted(enumerate(objects),key=lambda item:(item[1]['pixel_coords'][0],item[1]['pixel_coords'][1],item[0]))
    lines = ['Objects from left to right:']
    for number,(_,obj) in enumerate(ordered,1):
        lines.append(str(number)+'. '+' '.join(obj[key] for key in ['size','color','material','shape'])+'.')
    counts = Counter(obj['shape'] for obj in objects)
    lines.append('Shape counts: cubes '+str(counts['cube'])+', cylinders '+str(counts['cylinder'])+', spheres '+str(counts['sphere'])+'.')
    return '\n'.join(lines)


def prepare():
    source = ROOT/'data/clevr-reasoning-v1'; output = ROOT/'data/clevr-dense-scenes-v1'
    if output.exists(): raise ValueError('Existing dense scene preparation requires inspection; preserve prior artifacts')
    manifest = json.loads((source/'manifest.json').read_text()); audit = json.loads((ROOT/'reports/clevr-original-input-audit-v1.json').read_text())
    if audit.get('passed') is not True or audit['manifest_sha256'] != file_hash(source/'manifest.json'):
        raise ValueError('Original scene/image source audit differs')
    for name,digest in manifest['files_sha256'].items():
        if file_hash(source/name) != digest: raise ValueError('Original audited scene/question data changed')
    tokenizer_path = ROOT/'data/tokenizer-corpus/tokenizer.json'
    if file_hash(tokenizer_path) != manifest['tokenizer_sha256']: raise ValueError('Own tokenizer changed')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)),ModelConfig.load(ROOT/'configs/model.json'),output,448)
    output.mkdir(); stats = {}; files = {}; groups = {}
    for split in ['train','valid']:
        image_records = {}
        for row in jsonl(source/(split+'.jsonl')): image_records.setdefault(Path(row['image']).name,row)
        scenes = {scene['image_filename']:scene for scene in jsonl(source/'metadata'/(split+'-scenes.jsonl'))}
        rows = []; targets = 0
        for filename,original in image_records.items():
            scene = scenes[filename]; scene_sha = hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest()
            if scene_sha != original['scene_sha256']: raise ValueError('Dense scene differs from original question scene')
            image = (source/original['image']).resolve()
            if not image.is_relative_to(source.resolve()) or file_hash(image) != original['stored_image_sha256']:
                raise ValueError('Original dense-description image changed')
            row = {key:original[key] for key in ['group_id','source','source_url','license','license_url','attribution',
                   'stored_image_sha256','model_pixel_sha256','model_pixel_sha256_448','scene_sha256','original_zip_member']}
            row.update(id='clevr:dense:'+split+':'+filename,image=str(image),prompt=PROMPT,response=description(scene),
                       domain='visual_reasoning',task='dense_scene_description',source_subset=original['source_subset'],
                       supervision_origin='Our deterministic description of original public scene attributes/projected centers; no external model.',
                       external_model_teacher_used=False,synthetic_rendered_scene=True)
            ids,labels = encoder.sequence(row,'response')
            if encoder.truncated: raise ValueError('Complete original dense scene target exceeds own context')
            row['context_tokens']=len(ids);row['target_tokens']=sum(value!=-100 for value in labels[1:]);targets+=row['target_tokens'];rows.append(row)
        path=output/(split+'.jsonl');path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows));files[path.name]=file_hash(path)
        groups[split]={row['group_id'] for row in rows}
        stats[split]={'images':len(rows),'target_tokens':targets,'maximum_context_tokens':max(row['context_tokens'] for row in rows)}
    if groups['train'] & groups['valid']: raise ValueError('Dense original scene image groups overlap')
    report={'prepared_at':datetime.now(timezone.utc).isoformat(),'original_manifest_sha256':file_hash(source/'manifest.json'),
            'original_input_audit_sha256':file_hash(ROOT/'reports/clevr-original-input-audit-v1.json'),'tokenizer_sha256':manifest['tokenizer_sha256'],
            'splits':stats,'files_sha256':files,'implementation_sha256':file_hash(__file__),'trained':False,'goal_achieved':False,
            'limitations':['Original synthetic object attributes/center order; not real-image grounding or human feedback.',
                           'Extra target information and training budget must be disclosed in comparisons; no causal superiority claim from unmatched budgets.',
                           'Metadata generate training targets only; the model sees the image and identical task prompt, never scene graphs or gold attributes.']}
    atomic_json(output/'manifest.json',report);print('Original dense native scene supervision prepared',stats,flush=True)
    return report


if __name__=='__main__':prepare()
