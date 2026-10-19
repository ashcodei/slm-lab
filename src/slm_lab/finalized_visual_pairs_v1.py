"""Read complete admitted original paired photos; never admit a partial journal."""
from dataclasses import asdict
import json
from pathlib import Path

from .codenet_checkpoint_v1 import FileBindings
from .code_residual_collective_checkpoint_v2 import fingerprint
from .codenet_runner_v1 import VerifiedReplayEncoder
from .corpus import file_hash
from .paired_visual_contrast_v1 import pixel_digest
from .vqa_pair_admission import admit


class FinalizedVisualPairs:
    def __init__(self,root,tokenizer,config,image_size=448):
        self.root=Path(root).resolve();self.data=self.root/'data/vqa-real-pairs-v1'
        # The original full reconstruction is mandatory and rejects before
        # any partial image, row, token or GPU is consumed if unfinalized.
        rows,receipt=admit(self.root)
        if (receipt['passed'] is not True or receipt['validation_rows_not_returned_for_training'] is not True
                or receipt['all_original_question_answer_photo_pair_rows_reconstructed'] is not True):
            raise ValueError('Complete original paired-source admission required')
        self.manifest=json.loads((self.data/'manifest.json').read_text())
        from .model import ModelConfig
        if config!=ModelConfig.load(self.root/'configs/model.json') or type(image_size) is not int or image_size not in {224,448}:
            raise ValueError('Exact original own architecture and admitted224/448px inputs required')
        token=self.root/'data/tokenizer-corpus/tokenizer.json'
        from tokenizers import Tokenizer
        if file_hash(token)!=self.manifest['tokenizer_sha256'] or json.loads(tokenizer.to_str())!=json.loads(Tokenizer.from_file(str(token)).to_str()):
            raise ValueError('Exact own tokenizer bytes/semantics required')
        self.rows={r['id']:r for r in rows};self.pairs=list(map(json.loads,(self.data/'pairs.train.jsonl').read_text().splitlines()))
        if len(self.rows)!=len(rows) or not self.pairs:raise ValueError('Every original admitted training question/pair required')
        self.encoder=VerifiedReplayEncoder(tokenizer,config,self.data,image_size)
        self.tokenizer_json=json.loads(tokenizer.to_str())
        self.records_fingerprint=fingerprint(dict(rows=self.rows,pairs=self.pairs))
        implementation=[Path(__file__).resolve(),Path(__file__).with_name('paired_visual_contrast_v1.py'),
            Path(__file__).with_name('codenet_runner_v1.py')]
        self.contract=dict(format='own-finalized-original-visual-pair-contract-v1',receipt=receipt,config=asdict(config),
            tokenizer_sha256=file_hash(token),image_size=image_size,manifest_sha256=file_hash(self.data/'manifest.json'),
            original_records_fingerprint=self.records_fingerprint,
            source_population=dict(original_pairs=len(self.pairs),unique_original_questions=len(self.rows),
                original_question_membership_exposures=2*len(self.pairs)),
            implementation_sha256={str(p):file_hash(p) for p in implementation})
        self.contract_sha256=fingerprint(self.contract)
        files={**self.manifest['sources_sha256'],**{str(self.data/n):h for n,h in self.manifest['files_sha256'].items()},
            str(self.data/'manifest.json'):self.contract['manifest_sha256'],str(token):file_hash(token),
            str(Path(__file__).resolve()):file_hash(__file__)}
        files.update(self.contract['implementation_sha256'])
        for row in rows:
            for name,key in [('image','stored_image_sha256'),('attribution_page','attribution_page_sha256')]:
                files[str((self.data/row[name]).resolve())]=row[key]
        self.guard=FileBindings(files)

    def __len__(self):return len(self.pairs)

    def example(self,position):
        self.guard.check()
        if (fingerprint(self.contract)!=self.contract_sha256
                or fingerprint(dict(rows=self.rows,pairs=self.pairs))!=self.records_fingerprint
                or json.loads(self.encoder.tokenizer.to_str())!=self.tokenizer_json):
            raise ValueError('Every original source role/body and own tokenizer semantics must remain unchanged')
        if type(position) is not int or not 0<=position<len(self):raise IndexError('Exact original paired position required')
        pair=self.pairs[position];sides=[self.rows[key] for key in pair['question_ids']]
        if len(sides)!=2 or any(r['split']!='train' for r in sides) or sides[0]['prompt']!=sides[1]['prompt']:
            raise ValueError('Same complete original paired training question required')
        sequences=[];images=[]
        for row in sides:
            # Native VQA records keep the224px golden as the unsuffixed key;
            # the full replay encoder uses a resolution-suffixed field.
            encoded={**row,'model_pixel_sha256_224':row['model_pixel_sha256']}
            sequences.append(self.encoder.sequence(encoded,'response'))
            batch=self.encoder.batch([encoded],'response','cpu');images.append(batch['pixel_values'][0])
        if self.encoder.truncated:raise ValueError('Every original prompt/answer/EOS must remain whole')
        return dict(id=pair['id'],split='train',source='original_paired_VQA_human_answers',fully_finalized_source=True,
            source_contract_sha256=self.contract_sha256,human_preferences=False,external_model_teacher_used=False,
            question=pair['canonical_question'],answers=[r['response'] for r in sides],
            answer_types=[r['original_human_annotation']['answer_type'] for r in sides],human_votes=[r['exact_human_agreement'] for r in sides],
            photo_ids=[r['group_id'] for r in sides],pixel_sha256=[pixel_digest(x) for x in images],sequences=sequences,images=images)
