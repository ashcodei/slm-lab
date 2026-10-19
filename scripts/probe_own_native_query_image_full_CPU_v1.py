"""Real448px image-interface smoke on our complete own standalone model.

The deterministic color gradient has no semantic answer label and is not a
training/validation/benchmark image. This tests native interface execution
and exact original image preprocessing, not caption/OCR/model accuracy.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/own-native-query-image-full-CPU-v1'
BASE=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'
EXPECTED='759bcd549a6226a1cae00a5dd2eb0968355b445ad7d77629c84a6b3a9ec6e496'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden full-size image-interface smoke required')
    import numpy as np
    import torch
    from PIL import Image
    from tokenizers import Tokenizer
    from slm_lab.model import ModelConfig
    from slm_lab.training import Encoder
    from slm_lab.evaluation_provenance import pixels_sha256
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    text=ROOT/'reports/own-native-query-full-CPU-v1/complete.json'
    if not json.loads(text.read_text())['complete']:raise ValueError('Actual complete standalone text-interface proof required first')
    OUT.mkdir();fixture=OUT/'unlabelled-deterministic-color-gradient.png'
    values=np.arange(192*256*3,dtype=np.uint32).reshape(192,256,3)
    Image.fromarray((values%256).astype(np.uint8)).save(fixture)
    mapping=ROOT/'reports/own-native-query-v1-smoke-input-sources.json';token=BASE.parent/'tokenizer.json'
    paths=[BASE,token,mapping,text,fixture,*python_sources(ROOT,[Path(__file__),ROOT/'src/slm_lab/own_native_query_v1.py'])]
    files={str(p):file_hash(p) for p in paths}
    if files[str(BASE)]!=EXPECTED:raise ValueError('Exact original own complete model bytes required')
    guard=FileBindings(files);raw=torch.load(BASE,map_location='cpu',weights_only=True,mmap=True)
    config=ModelConfig(**raw['complete_dense']['config']);encoder=Encoder(Tokenizer.from_file(str(token)),config,OUT,448)
    expected_pixels_sha256=pixels_sha256(encoder.pixels({'image':str(fixture)}))
    del raw
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,CPU_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),actual_image_size=448,
        expected_original_Encoder_pixel_sha256=expected_pixels_sha256,expected_visual_tokens=config.image_token_count(448),
        no_semantic_expected_answer_or_quality_score=True,original_benchmark_or_training_image_not_used=True,
        no_GPU_allocation_or_training_updates=True,original_jobs_unmodified=True,goal_achieved=False))
    command=[sys.executable,'-m','slm_lab.own_native_query_v1','--checkpoint',str(BASE),'--expected-sha256',EXPECTED,
        '--expected-tokenizer-sha256',files[str(token)],'--expected-sources',str(mapping),
        '--prompt','Describe the image.','--image',str(fixture),'--image-size','448','--max-new-tokens','1','--threads','1','--precision','bf16','--json']
    start=time.perf_counter()
    with (OUT/'actual-query.json').open('x') as output,(OUT/'stderr.log').open('x') as errors:
        subprocess.run(command,cwd=ROOT,stdout=output,stderr=errors,check=True)
    actual=json.loads((OUT/'actual-query.json').read_text())
    if (actual['actual_registered_parameters']!=984399690 or actual['actual_native_actions']!=1
            or actual['checkpoint_sha256']!=EXPECTED or actual['inference_device']!='cpu'
            or actual['image']['pixel_tensor_sha256']!=expected_pixels_sha256 or actual['image']['image_size']!=448
            or actual['image']['visual_prefix_tokens']!=config.image_token_count(448) or torch.cuda.is_initialized()):
        raise ValueError('Actual full model image interface must match the exact original448px pixels and complete native action')
    guard.check(full=True)
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_registered_neural_parameters=984399690,
        actual_image_size=448,actual_visual_tokens=config.image_token_count(448),actual_native_actions=1,
        original_Encoder_and_frontend_image_preprocessing_bit_exact=True,actual_complete_CPU_BF16_image_conditioned_generation=True,
        end_to_end_seconds=time.perf_counter()-start,actual_generation_seconds=actual['generation_seconds'],
        no_new_generated_quality_metric=True,no_new_training_updates=True,no_original_job_modified=True,
        source_files_sha256={**files,str(OUT/'plan.json'):file_hash(OUT/'plan.json'),str(OUT/'actual-query.json'):file_hash(OUT/'actual-query.json')},goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_neural_parameters=984399690,image_size=448,no_quality_metric=True)),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        target=OUT/'failure.json' if OUT.exists() else ROOT/'reports/own-native-query-image-full-CPU-v1-parent-failure.json'
        atomic_json(target,dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise
