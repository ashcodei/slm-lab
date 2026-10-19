# SLM Lab

A sub-billion parameter multimodal language model built from scratch focusing on coding, general text, and image understanding.

- An own text decoder and vision encoder, with a **984M-parameter standalone model** and a shared 32K-token vocabulary.
- Public text, code, image, and OCR data pipelines with source tracking and validation. A completed training stage covered **1.106B response/EOS targets across 17 source families**.
- Post-training implementations for supervised fine-tuning, human-preference DPO, reward modeling/PPO experiments, custom LoRA-style adapters, and execution-guided code RL.
- Distributed training on **four NVIDIA L40 GPUs**, resumable optimizer/RNG checkpoints, and standalone text/image inference.
- Evaluation for code syntax, functional correctness and runtime efficiency, alongside knowledge, math, OCR, and visual question answering.

Built with **Python, PyTorch, CUDA, torch.distributed (NCCL/Gloo), Hugging Face Datasets/Tokenizers, Pillow, and pytest**. The backbone originated from random initialization; released synthetic coding data is tracked separately from human-written code and human feedback.

## Results so far

A completed public-data stage reduced reserved-code token loss from **0.8193 to 0.7897**, a **3.60% reduction**. In a separate matched code-repair experiment:

| Local evaluation | Baseline | Semantic repair |
|---|---:|---:|
| Syntactically valid programs | 14/48 | 30/48 |
| Functionally correct programs | 1/48 | 1/48 |
| Programs passing efficiency checks | 0/48 | 1/48 |
| Python functional correctness | 0/24 | 1/24 |

On **4,642 held human-ranked pairs**, DPO reached **54.85% reference-relative preference accuracy**, versus **50.78%** for chosen-response SFT. Baseline and DPO coding correctness both remained **1/48**.

Correctness and broad multimodal performance remain limited. These are small local development suites, not leaderboard results; training and evaluation are ongoing.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/ashcodei/slm-lab.git
cd slm-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[data,dev,eval]'
```

Run the small CPU model checks:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q tests/test_model.py
```

## Run

For inference, use an exported standalone `model.pt` with its matching `tokenizer.json` in the same directory:

```bash
CUDA_VISIBLE_DEVICES='' python -m slm_lab.own_native_query_v1 \
  --checkpoint path/to/model.pt \
  --prompt "Write a Python function to add two integers." \
  --max-new-tokens 128
```

Add `--image path/to/image.png --image-size 448` for an image question, or `--raw-prompt` for code completion.

For an unfused policy/reward package, use `python -m slm_lab.own_policy_reward_query_v1 --checkpoint path/to/model.pt --prompt "Your question"`. Add `--score-response "Your response"` to obtain its text reward score.

The training CLI supports `pretrain`, `sft`, `contrastive`, `dpo`, `reward`, and `ppo`:

```bash
python -m slm_lab.training --help
```

Datasets, model weights, environments, and generated reports stay local and are excluded from Git. Training requires prepared data and a matching tokenizer. Experiment drivers are in `scripts/`; detailed research history is in [RESEARCH.md](RESEARCH.md).
