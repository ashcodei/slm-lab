"""Report host hardware separately from usable accelerators; never allocate devices."""
import json
import os
from pathlib import Path
import torch


def inspect():
    host=[]
    for path in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        try:
            lines=path.read_text().splitlines()
        except OSError:continue
        model=next((line.split(':',1)[1].strip() for line in lines if line.startswith('Model:')),None)
        if model:host.append(model)
    accessible=[str(p) for p in Path('/dev').glob('nvidia[0-9]*')]
    usable=torch.cuda.is_available()
    report={'host_gpu_models':host,'device_nodes':accessible,
        'nvidia_visible_devices':os.environ.get('NVIDIA_VISIBLE_DEVICES'),
        'torch_version':torch.__version__,'torch_cuda_build':torch.version.cuda,
        'cuda_available':usable,'usable_gpu_count':torch.cuda.device_count() if usable else 0,
        'allocation_available':bool(accessible),
        'interpretation':'Host inventory is not proof of allocation, availability, or permission.'}
    if usable:
        report['usable_gpus']=[{'name':torch.cuda.get_device_name(i),
            'vram_bytes':torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())]
    if usable:report['next_action']='CUDA is usable; benchmark the model before a sustained run.'
    elif not accessible:report['next_action']='Select a GPU-enabled profile in the launcher or ask its administrator.'
    elif torch.version.cuda is None:report['next_action']='Install a CUDA PyTorch build compatible with the exposed driver.'
    else:report['next_action']='CUDA initialization needs diagnosis; do not infer availability from /proc alone.'
    return report


def main():print(json.dumps(inspect(),indent=2))


if __name__=='__main__':main()
