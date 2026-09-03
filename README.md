# DC2-PET: Dual Control Diffusion with Nuclear Regularization for Ultra-low-dose PET Imaging

This is the official implementation of **DC2-PET**. A conditional diffusion model for ultra-low-dose PET imaging that recovers standard-dose quality from ultra-low-dose (e.g., 4%–10% dose) whole-body PET acquisitions with minimal loss of quantitative accuracy and image quality.

> Paper: DC2-PET: Dual Control Diffusion with Nuclear Regularization for Ultra-low-dose PET Imaging
> Authors / publication info: to be added | [Paper link](#) | [Pretrained weights](#open-source-weights)

---

## Table of Contents

- [Method Overview: DC2](#method-overview-dc2)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Dataset Organization](#dataset-organization)
- [Open-Source Weights](#open-source-weights)
- [Weight Merging Tools](#weight-merging-tools)
- [Training](#training)
- [Sampling and Evaluation](#sampling-and-evaluation)
- [Ablation: Single-Control Model (NoneLSCC)](#ablation-single-control-model-nonelscc)
- [Citation](#citation)

---

## Method Overview: DC2

DC2 is a standard-dose PET diffusion generator conditioned on **low-dose PET images**. The core architecture is defined in `improved_diffusion/LSCC.py` (the overall model `LowRankSparseCompressControlNet`) and `improved_diffusion/CDM.py` (ControlNet / ControlIddpm). It comprises **two control pathways**:

1. **Image-level structural control (ControlNet branch)**
   The low-dose PET image (`hint`) is encoded by `input_hint_block` into multi-scale guidance features that are injected into the denoising U-Net at every scale through zero-convolutions (`ControlNetWithLSCT` / `ControlNet`), constraining generation to align with the anatomy and lesion locations in the low-dose input. The main denoising backbone is a pretrained standard-dose diffusion U-Net (`ControledUnetModel`, which can be frozen via `sd_locked` so that only the control branch is trained).

2. **Semantic-level control (Low-rank Sparse Compress Vector, LSCV)**
   A **frozen** dose-aware feature encoder (classifier) extracts an **LSCV** (**L**ow-rank **S**parse **C**ompress **V**ector, 768-d) from the low-dose input. The vector is gated through `LSCV_embed` + a **zero-initialized linear** layer and added to the diffusion timestep embedding, modulating denoising according to the dose level. The default encoder is a low-rank sparse compression Transformer (`LSCClassifier`, see `improved_diffusion/LSCT.py`), whose ISTA / soft-thresholding-style proximal layers perform low-rank sparse compression of the features — corresponding to the nuclear regularization in the paper.

The two control pathways jointly let the model exploit the global semantic of *how severely the low-dose input has degraded* to guide diffusion sampling, producing high-quality reconstruction from ultra-low-dose to standard-dose.

**Encoder variants**: the feature encoder can be switched via `--feType`. Three variants are released (default `LSCC`):

| feType   | Feature encoder                  | Notes                                 |
| -------- | -------------------------------- | ------------------------------------- |
| `LSCC`   | `LSCClassifier_Base`             | Default: Low-rank Sparse Compress Transformer (LSCT) |
| `ViT`    | `ViTClassifier_Base_p16`         | ViT-Base/16 encoder variant           |
| `ResNet` | `ResNetClassifier_base16`        | ResNet encoder variant                |

All three variants share the same DC2 dual-control architecture and differ only in the frozen feature encoder.

---

## Repository Structure

```
.
├── image_train_LSCC.py            # DC2 training entry (supports --feType LSCC/ViT/ResNet)
├── image_sample_LSCC.py           # DC2 sampling entry (saves images and computes PSNR)
├── image_train_control.py         # Single-control (NoneLSCC, no LSCV) ablation training
├── image_sample_control.py        # Single-control (NoneLSCC) ablation sampling
├── tool_add_control.py            # Weight merging tool 1: diffusion backbone + ControlNet branch
├── tool_add_control_and_classifier.py  # Weight merging tool 2: merges the full DC2 weights
├── pre_train_iddpm/               # Open-source weights directory
├── improved_diffusion/            # Core code
│   ├── LSCC.py                    #   DC2 overall model: LowRankSparseCompressControlNet
│   ├── CDM.py                     #   ControlNet / ControlNetWithLSCT / ControlIddpm
│   ├── LSCT.py                    #   Low-rank sparse compress Transformer classifier (default encoder)
│   ├── ViTC.py / ResNetC.py       #   ViT / ResNet encoder variants
│   ├── unet.py                    #   Base diffusion U-Net backbone (iddpm)
│   ├── script_util.py             #   Model construction and defaults for each variant
│   ├── image_datasets.py          #   DICOM → SUV data loading
│   ├── gaussian_diffusion.py      #   Diffusion process / sampling
│   └── train_util*.py             #   Training loops
├── LSCT/                          # Standalone pretraining code for feature encoders (LSCClassifier/ViT/ResNet)
├── grad_cam.py                    # Visualization tool
├── Configs/                       # History of experiment commands (reference examples)
└── setup.py
```

---

## Environment Setup

```bash
# Python 3.8+ / PyTorch (CUDA build recommended)
pip install -e .
# Additional dependencies used by the code (install as needed)
pip install mpi4py blobfile einops pydicom scikit-image matplotlib torchvision tqdm
```

The train/sample scripts use OpenAI improved-diffusion-style MPI wrappers (`dist_util.setup_dist()`). For single-GPU runs, simply `python xxx.py` (equivalent to a single-process MPI run).

> ⚠️ Note: the logging directories (`logger.configure(dir=...)`) and GPU selection (`CUDA_VISIBLE_DEVICES`) in `image_train_LSCC.py` and `image_sample_LSCC.py` are **hard-coded absolute paths**. Modify them before running.

---

## Dataset Organization

`image_datasets.py` reads **DICOM** directly and expects data organized by patient directory and dose directory:

```
<data_dir>/
├── <patient_xxx>/
│   ├── 2.886 x 600 WB NORMAL/     # Standard-dose ground truth
│   │   └── slice_xxxx.dcm
│   ├── 2.886 x 600 WB D100/       # Low-dose input, dose levels D100/D50/D20/D10/D4
│   ├── 2.886 x 600 WB D50/
│   └── ...
└── ...
```

- The loader converts pixel values to **SUV** using DICOM header information (see `get_SUV`), then pads/crops to `256 × 256`.
- `--dose` options: `D100` / `D50` / `D20` / `D10` / `D4` / `ALL` (mixed-dose training) / `G10` (dedicated layout reading `<data_dir>/G10/` and `<data_dir>/G180/`).
- For training we recommend `--dose=ALL` (joint mixed-dose training); for sampling, pass the target dose via `--dose`.

---

## Open-Source Weights

Pretrained weights for the DC2 variants are released as **GitHub Release assets** (they exceed GitHub's per-file limit for regular git pushes and are therefore not stored in the repository tree; the local [`pre_train_iddpm/`](./pre_train_iddpm/) folder is gitignored).

Download the released weights from the [**pretrained-weights**](https://github.com/yqx7150/DCDM/releases/tag/pretrained-weights) release:

| Weight file                                        | Corresponding model / feType  | Purpose                                        |
| -------------------------------------------------- | ----------------------------- | ---------------------------------------------- |
| [model30W_LSCC30W.pt](https://github.com/yqx7150/DCDM/releases/download/pretrained-weights/model30W_LSCC30W.pt) | DC2, `--feType=LSCC` (default)| LSCT low-rank sparse compress encoder variant  |
| `model30W_ViT10W.pt` / `model30W_ResNet20W.pt` / `model30W.pt` | —                     | Coming soon                                   |

The DC2 weights can be passed either as `--resume_checkpoint` to `image_train_LSCC.py` (fine-tuning from them) or directly as `--model_path` to `image_sample_LSCC.py` for sampling. The `30W` / `20W` / `10W` suffixes in the filenames indicate the magnitude of iterations (300k / 200k / 100k) used for training or merging.

---

## Weight Merging Tools

After adding the control branch / feature encoder, the model parameter names no longer match those of the base diffusion backbone, so pretrained weights cannot be loaded with a plain `load_state_dict`. The two tools **merge separately-pretrained parts into a complete weight file** by remapping and copying keys:

### 1. `tool_add_control.py` — Merge "base diffusion backbone + ControlNet branch"

Applies to the **single-control model without LSCV** (NoneLSCC, i.e., the ablation model used by `image_train_control.py` / `image_sample_control.py`, with `insert_LSCV=False`). It strips the `control_` prefix from control-branch keys, looks up the corresponding layers in the base backbone weights, loads the base diffusion weights into the backbone (control-branch keys prefixed with `controled_` are remapped accordingly), and leaves newly-added layers randomly initialized.

```bash
python tool_add_control.py \
    --input_path=<pretrained base diffusion weights, e.g., ./pre_train_iddpm/model30W.pt> \
    --output_path=<output path for the merged weights>
```

### 2. `tool_add_control_and_classifier.py` — Merge "diffusion backbone + control branch + feature encoder" into the full DC2

Applies to the complete **DC2** model (`LowRankSparseCompressControlNet`, `insert_LSCV=True`). It simultaneously:

- remaps the pretrained feature-encoder (LSCT / ViT / ResNet) weights into the `classifier.*` part;
- remaps the pretrained diffusion backbone and control-branch weights into the `controledIddpm.*` part.

```bash
python tool_add_control_and_classifier.py \
    --classifier_path=<pretrained feature-encoder weights (trained with the LSCT/ code)> \
    --diffusion_path=<pretrained diffusion backbone weights, e.g., ./pre_train_iddpm/model30W.pt> \
    --output_path=<output path for the merged DC2 weights>
```

> The released `pre_train_iddpm/model30W_*` weights are exactly the output of this tool. In most cases you do not need to merge again — you can jump straight to [Training](#training) or [Sampling and Evaluation](#sampling-and-evaluation).

---

## Training

Train DC2 with `image_train_LSCC.py`. Choose the variant via `--feType` and initialize with the corresponding weights:

```bash
# LSCC (default feature encoder)
python image_train_LSCC.py \
    --data_dir=<training data root> \
    --class_cond=False \
    --dose="ALL" \
    --iteration=120001 \
    --batch_size=1 \
    --resume_checkpoint="./pre_train_iddpm/model30W_LSCC30W.pt"

# ViT variant
python image_train_LSCC.py \
    --data_dir=<training data root> \
    --class_cond=False --dose="ALL" \
    --iteration=120001 --batch_size=1 \
    --feType="ViT" \
    --resume_checkpoint="./pre_train_iddpm/model30W_ViT10W.pt"

# ResNet variant
python image_train_LSCC.py \
    --data_dir=<training data root> \
    --class_cond=False --dose="ALL" \
    --iteration=120001 --batch_size=1 \
    --feType="ResNet" \
    --resume_checkpoint="./pre_train_iddpm/model30W_ResNet20W.pt"
```

Default model settings (`256×256`, `num_channels=128`, `attention_resolutions=32,16,8`, `LSCV_dim=768`, `num_doses=5`, etc.) are defined in `improved_diffusion/script_util.py` by `LSCC_and_diffusion_defaults()` / `ViT_and_diffusion_defaults()` / `ResNet_and_diffusion_defaults()`.

---

## Sampling and Evaluation

Generate reconstructions with `image_sample_LSCC.py`. For every sample the script computes **PSNR** and saves `sample` / `label` (ground truth) / `input` (low-dose input) as `.npz` files in the directory set by `logger.configure(dir=...)`.

```bash
# LSCC (default) variant
python image_sample_LSCC.py \
    --data_dir=<evaluation data root> \
    --class_cond=False \
    --dose="D20" \
    --batch_size=1 \
    --num_samples=672 \
    --model_path="./pre_train_iddpm/model30W_LSCC30W.pt"

# ViT variant
python image_sample_LSCC.py \
    --data_dir=<evaluation data root> \
    --class_cond=False --dose="D20" \
    --batch_size=1 --num_samples=672 \
    --feType="ViT" \
    --model_path="./pre_train_iddpm/model30W_ViT10W.pt"

# ResNet variant
python image_sample_LSCC.py \
    --data_dir=<evaluation data root> \
    --class_cond=False --dose="D20" \
    --batch_size=1 --num_samples=672 \
    --feType="ResNet" \
    --model_path="./pre_train_iddpm/model30W_ResNet20W.pt"
```

- Pass the evaluation dose via `--dose`. Metrics such as SSIM / PSNR can be further computed from the output `.npz` files; the `convertion/` directory keeps historical metric-computation logs for reference.
- Add `--use_ddim` for accelerated DDIM sampling.

---

## Ablation: Single-Control Model (NoneLSCC)

The single-control ablation in the paper (removing the LSCV semantic control and keeping only the image-level structural control) uses `image_train_control.py` / `image_sample_control.py`. Generate the weights with `tool_add_control.py`, then train and sample:

```bash
# 1) Merge the base weights into the single-control model
python tool_add_control.py \
    --input_path="./pre_train_iddpm/model30W.pt" \
    --output_path=<output path>

# 2) Training
python image_train_control.py \
    --data_dir=<training data root> \
    --class_cond=False --dose="ALL" \
    --iteration=120001 --batch_size=1 \
    --resume_checkpoint=<merged weights from the previous step>

# 3) Sampling
python image_sample_control.py \
    --data_dir=<evaluation data root> \
    --class_cond=False --dose="D20" \
    --batch_size=1 --num_samples=672 \
    --model_path=<trained model>
```

---

## Citation

If this code or the DC2-PET method helps your research, please cite our paper (to be added):

```bibtex
@article{dc2pet,
  title={DC2-PET: Dual Control Diffusion with Nuclear Regularization for Ultra-low-dose PET Imaging},
  author={},
  year={},
}
```
