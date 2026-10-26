# NTI: Next Token Imagination

Training and evaluation code for **Visual Autoregressive Generation with Next
Token Imagination** on ImageNet 256×256.

Ruihang Jiang, Zongyao Zhao, Zhaolin Wang, Yuanwei Liu, and Xiatian Zhu.

The arXiv preprint link will be added once available.

The Python package is `nti`; the training and sampling entry points are
`scripts/train_nti.py` and `scripts/sample_imagenet_nti.py`.

## Setup

Use Python 3.11 and a CUDA-compatible PyTorch installation.

```bash
git clone https://github.com/JackRuihang/NTI.git
cd NTI
pip install -e '.[data,test]'
git clone https://github.com/QY-H00/reAR external/reAR
git -C external/reAR checkout 61cc871cc188e4593160156480a10b052eececdb
```

Download the MaskGIT ImageNet-256 f16 tokenizer from the
[reAR repository](https://github.com/QY-H00/reAR) and save it as
`assets/tokenizer.bin`. ImageNet must be obtained separately.
Asset checksums are listed in [ASSET_CHECKSUMS.md](ASSET_CHECKSUMS.md).

## Checkpoints

The NTI-S/B/L 250K-step generator inference checkpoints are available in
[Releases](https://github.com/JackRuihang/NTI/releases/tag/v0.1.0), separately from
the Git source tree, under the MIT License.

| Model | Generator parameters | Download |
|---|---:|---|
| NTI-S | 218,396,416 | [nti_s_250k.pt](https://github.com/JackRuihang/NTI/releases/download/v0.1.0/nti_s_250k.pt) |
| NTI-B | 260,922,880 | [nti_b_250k.pt](https://github.com/JackRuihang/NTI/releases/download/v0.1.0/nti_b_250k.pt) |
| NTI-L | 461,667,328 | [nti_l_250k.pt](https://github.com/JackRuihang/NTI/releases/download/v0.1.0/nti_l_250k.pt) |

Download and verify the files before sampling:

```bash
mkdir -p weights
cd weights
curl -fL -o nti_s_250k.pt https://github.com/JackRuihang/NTI/releases/download/v0.1.0/nti_s_250k.pt
curl -fL -o nti_b_250k.pt https://github.com/JackRuihang/NTI/releases/download/v0.1.0/nti_b_250k.pt
curl -fL -o nti_l_250k.pt https://github.com/JackRuihang/NTI/releases/download/v0.1.0/nti_l_250k.pt
curl -fL -o SHA256SUMS.txt https://github.com/JackRuihang/NTI/releases/download/v0.1.0/SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
cd ..
```

The files are unchanged inference exports containing model weights, configuration,
and training step. They do not contain optimizer or replay state and cannot resume
training. The frozen tokenizer is not included and must be obtained separately as
described above. File sizes and hashes are also provided in the release's
`checkpoint_manifest.json`.

## Train

Prepare ImageNet tokens, including horizontal flips:

```bash
torchrun --standalone --nproc_per_node=8 scripts/pretokenize_imagenet.py \
  --image-root data/imagenet/train --checkpoint assets/tokenizer.bin \
  --rear-repo external/reAR --output-dir data/tokens --views original,hflip
```

Set the token manifest path in `configs/nti_s.yaml`, then run:

```bash
python scripts/run_training.py --config configs/nti_s.yaml --nproc 8 --bank-batch-size 64
```

Use `nti_b.yaml` or `nti_l.yaml` for the other model sizes.
Training uses 250K steps and global batch size 2,048. The launcher builds and
refreshes the replay bank every 10K steps, starting at 10K. The last 50K steps
use clean training. Use a new output and replay directory for each run.

To resume at a completed phase boundary, add `--resume-step N` and retain the
training checkpoint and its replay bank. Inference exports cannot resume training.
Within a phase, use `scripts/train_nti.py --resume` with the full training checkpoint,
the unchanged configuration, and the matching replay bank. Do not substitute a bank
from another snapshot or change `--max-steps` to stop early: use `--stop-at-step`.

## Evaluate

Generate 50K images (NTI-L example):

```bash
torchrun --standalone --nproc_per_node=8 scripts/sample_imagenet_nti.py \
  --config configs/nti_l.yaml --checkpoint weights/nti_l_250k.pt \
  --output samples/nti_l_seed42 --num-samples 50000 --batch-size 50 \
  --seed 42 --guidance-scale 14 --guidance-scale-pow 1.75 \
  --temperature 1 --top-k 0 --top-p 1
```

| Model | CFG | CFG power |
|---|---:|---:|
| NTI-S | 22 | 2.75 |
| NTI-B | 14.5 | 2.25 |
| NTI-L | 14 | 1.75 |

Use sampling seeds 42, 142 and 242 for the three-seed results.
Temperature is 1, without top-k or top-p truncation.

Install the [ADM evaluator](https://github.com/openai/guided-diffusion/tree/main/evaluations)
in its own environment and download its ImageNet-256 reference batch.
Pass the NPZ path printed by the sampler to:

```bash
python scripts/run_adm_evaluator.py \
  --evaluator external/guided-diffusion/evaluations/evaluator.py \
  --evaluator-python /path/to/adm-env/bin/python \
  --samples /path/to/generated_samples.npz \
  --expected-num-samples 50000 --expected-image-size 256 \
  --reference ImageNet256=assets/VIRTUAL_imagenet256_labeled.npz \
  --output metrics/nti_l_seed42.json
```

## Tests

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests
```

Use the provided YAML configurations and prepare token/replay caches with the
scripts above. The tokenizer and ADM evaluator retain their original licenses.

## License

Our original NTI contributions and released NTI-S/B/L generator checkpoints are
provided under the [MIT License](LICENSE), which permits commercial use.
Third-party components and assets retain their original terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
