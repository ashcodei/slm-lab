"""Whole-query 448px inputs, with original roles and separate training admission."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tokenizers import Tokenizer
import torch

from .corpus import file_hash
from .localized_pointer_evidence_v1 import dense_target


class PointerData:
    def __init__(self, root, role, purpose, *, proof=None, tokenizer=None):
        if role not in ('train','validation') or purpose not in ('optimization','evaluation','prior','audit'):
            raise ValueError('Explicit original source role and input purpose required')
        if (purpose in ('optimization','prior') and role != 'train') or (purpose == 'evaluation' and role != 'validation'):
            raise ValueError('Validation never supplies gradients or fitted position priors')
        self.root = Path(root).resolve(); self.role = role; self.purpose = purpose
        project = Path(__file__).resolve().parents[2]
        proof = Path(proof) if proof is not None else project/'reports/localized-pointer-candidate-v1/independent-source-readback.json'
        token = Path(tokenizer) if tokenizer is not None else project/'checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json'
        manifest_path = self.root/'manifest.json'
        self.manifest = json.loads(manifest_path.read_text())
        evidence = json.loads(proof.read_text())
        if (evidence['candidate_manifest_sha256'] != file_hash(manifest_path)
                or not evidence['every_whole_original_query_source_record_offset_sha_times_and_point_index_reconstructed']
                or not evidence['every_actual_photo_header_EXIF_dimensions_pixels_creator_and_role_rechecked']
                or not evidence['every_query_BPE_BOS_content_EOS_and_448px_14x14_grid_count_reconstructed']):
            raise ValueError('Actual complete independent source reconstruction required')
        for name,sha in self.manifest['files_sha256'].items():
            if file_hash(self.root/name) != sha:
                raise ValueError('Whole original query/photo manifest bytes changed')
        if file_hash(token) != self.manifest['tokenizer_sha256']:
            raise ValueError('Exact own native BPE required')
        if purpose == 'optimization':
            admission = json.loads((self.root/'admission.json').read_text())
            if (not admission.get('training_admitted') or admission['scope'] != 'original-human-attention-distribution-supervision-only'
                    or admission['independent_source_readback_sha256'] != file_hash(proof)
                    or admission['candidate_manifest_sha256'] != file_hash(manifest_path)
                    or admission['original_validation_optimization_forbidden'] is not True):
                raise ValueError('Separate exact train-only original-attention admission required')
        self.tokenizer = Tokenizer.from_file(str(token))
        self.rows = [json.loads(line) for line in (self.root/(role+'.jsonl')).read_text().splitlines()]
        self.photos = {}
        for photo in map(json.loads,(self.root/'photos.jsonl').read_text().splitlines()):
            if photo['split'] == role:
                if photo['image_id'] in self.photos:
                    raise ValueError('One exact original photo identity per source role required')
                self.photos[photo['image_id']] = photo
        seen = set()
        for row in self.rows:
            if (row['id'] in seen or row['split'] != role or row['optimization_forbidden'] != (role != 'train')
                    or row['image_id'] not in self.photos or row['image_size'] != 448 or row['grid_side'] != 14
                    or row['query_input_ids'] != [1,*self.tokenizer.encode(row['query']).ids,2]
                    or not 3 <= len(row['query_input_ids']) <= 34):
                raise ValueError('Complete unmodified original query, role and own-token sequence required')
            seen.add(row['id'])
            dense_target(row,self.photos[row['image_id']]['stored_dimensions'])
        if len(self.rows) != self.manifest['complete_query_rows'][role] or not self.rows:
            raise ValueError('Every original admitted query row required')

    def __len__(self):
        return len(self.rows)

    def pixels(self, image_id):
        photo = self.photos[image_id]
        if file_hash(photo['image']) != photo['stored_image_sha256']:
            raise ValueError('Actual retained original photo bytes changed')
        with Image.open(photo['image']) as image:
            image = ImageOps.pad(image.convert('RGB'),(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128))
            import hashlib
            if hashlib.sha256(image.tobytes()).hexdigest() != photo['model_pixel_sha256_448']:
                raise ValueError('Actual exact 448px model input differs from original proof')
            values = np.asarray(image,dtype=np.float32)/127.5-1
        return torch.from_numpy(values.copy()).permute(2,0,1)

    def batch(self, positions, device='cpu'):
        if not positions or any(type(p) is not int or not 0 <= p < len(self.rows) for p in positions):
            raise ValueError('Actual whole-query source positions required')
        rows = [self.rows[p] for p in positions]
        length = max(len(r['query_input_ids']) for r in rows)
        ids = torch.zeros(len(rows),length,dtype=torch.long)
        mask = torch.zeros_like(ids,dtype=torch.bool)
        targets = []; valid = []
        for i,row in enumerate(rows):
            query = row['query_input_ids']; ids[i,:len(query)] = torch.tensor(query); mask[i,:len(query)] = True
            counts,cells = dense_target(row,self.photos[row['image_id']]['stored_dimensions'])
            targets.append(counts); valid.append(cells)
        return dict(input_ids=ids.to(device), attention_mask=mask.to(device),
            pixel_values=torch.stack([self.pixels(r['image_id']) for r in rows]).to(device),
            targets=torch.tensor(targets,dtype=torch.float32,device=device),
            valid_cells=torch.tensor(valid,dtype=torch.bool,device=device))
