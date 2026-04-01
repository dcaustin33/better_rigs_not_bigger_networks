# Better Rigs, Not Bigger Networks: A Body Model Ablation for Gaussian Avatars

**Anonymous CVPR 2026 Submission**

Recent 3D Gaussian splatting methods built atop SMPL achieve remarkable visual fidelity while continually increasing the complexity of the overall training architecture. We demonstrate that much of this complexity is unnecessary. By replacing SMPL with the Momentum Human Rig (MHR), estimated via SAM-3D-Body, a minimal pipeline with no learned deformations or pose-dependent corrections achieves the highest reported PSNR and competitive SSIM and LPIPS on PeopleSnapshot and ZJU-MoCap. To disentangle pose estimation quality from body model representational capacity, we perform controlled ablations: translating SAM-3D-Body meshes back to SMPL-X, and translating the original dataset's SMPL poses into MHR, both retrained under identical conditions. These ablations confirm that body model expressiveness has been a primary bottleneck in accurate reconstruction, with both mesh representational capacity and pose estimation quality contributing meaningfully to the full pipeline's gains.

Key findings:
- A minimal Gaussian avatar pipeline that, by substituting MHR for SMPL and removing all learned deformation modules, achieves the highest reported PSNR on both PeopleSnapshot and ZJU-MoCap.
- A controlled ablation that translates poses between body models under identical training conditions, disentangling the contributions of pose estimation quality and mesh representational capacity.
- MHR's denser mesh (18,439 vertices at LOD 1 vs. 6,890 for SMPL) places Gaussians closer to the true surface, producing geometrically more accurate renderings before any learned correction has a chance to act.

## Installation

```bash
# Install dependencies using uv
uv sync
```

For CUDA support (gsplat, fused-ssim, KNN_CUDA):
```bash
uv sync --extra cuda
```

## Model Setup

| Model | Download URL | Place in |
|-------|-------------|----------|
| SMPL | https://smpl.is.tue.mpg.de | `mesh_models/smpl/models/` |
| SMPL-X | https://smpl-x.is.tue.mpg.de | `mesh_models/models/smplx/` |
| MHR | See `scripts/setup_mhr_assets.py` | `mesh_models/mhr/assets/` |

## Datasets

- **PeopleSnapshot** (corrected format): Place in `datasets/people_snapshot_corrected/<subject>/`
- **ZJU MoCap**: Place in `datasets/zju_mocap/<subject>/`

## Training

### PeopleSnapshot (MHR)

```bash
uv run python scripts/train.py --config configs/male-3-casual-mhr.yaml
uv run python scripts/train.py --config configs/female-3-casual-mhr.yaml
uv run python scripts/train.py --config configs/male-4-casual-mhr.yaml
uv run python scripts/train.py --config configs/female-4-casual-mhr.yaml
```

### PeopleSnapshot (SMPL-X)

```bash
uv run python scripts/train.py --config configs/male-3-casual-smplx.yaml
uv run python scripts/train.py --config configs/female-3-casual-smplx.yaml
uv run python scripts/train.py --config configs/male-4-casual-smplx.yaml
uv run python scripts/train.py --config configs/female-4-casual-smplx.yaml
```

### PeopleSnapshot (SMPL-to-MHR converted)

```bash
uv run python scripts/train.py --config configs/male-3-casual-smpl2mhr.yaml
uv run python scripts/train.py --config configs/female-3-casual-smpl2mhr.yaml
```

### ZJU MoCap (MHR)

```bash
uv run python scripts/train.py --config configs/zju-377-mhr.yaml
uv run python scripts/train.py --config configs/zju-386-mhr.yaml
# ... similarly for 387, 392, 393, 394
```

### ZJU MoCap (SMPL-X)

```bash
uv run python scripts/train.py --config configs/zju-377-smplx.yaml
uv run python scripts/train.py --config configs/zju-386-smplx.yaml
# ... similarly for 387, 392, 393, 394
```

## Evaluation

```bash
uv run python scripts/evaluate.py --config configs/eval-template.yaml \
    --checkpoint output/<experiment>/checkpoints/iter_30000.pt
```

## Inference

```bash
# Single frame
uv run python scripts/inference.py --checkpoint output/<experiment>/final.pt \
    --pose-file pose.npz --device mps

# Batch
uv run python scripts/inference.py --checkpoint output/<experiment>/final.pt \
    --pose-dir poses/ --output output/batch/
```

## Visualization

```bash
uv run python scripts/visualize.py --checkpoint output/<experiment>/final.pt
```

## Available Configs

### PeopleSnapshot
| Config | Subject | Model |
|--------|---------|-------|
| `male-3-casual-mhr.yaml` | male-3-casual | MHR |
| `male-4-casual-mhr.yaml` | male-4-casual | MHR |
| `female-3-casual-mhr.yaml` | female-3-casual | MHR |
| `female-4-casual-mhr.yaml` | female-4-casual | MHR |
| `male-3-casual-smplx.yaml` | male-3-casual | SMPL-X |
| `male-4-casual-smplx.yaml` | male-4-casual | SMPL-X |
| `female-3-casual-smplx.yaml` | female-3-casual | SMPL-X |
| `female-4-casual-smplx.yaml` | female-4-casual | SMPL-X |
| `male-3-casual-smpl2mhr.yaml` | male-3-casual | SMPL→MHR |
| `female-3-casual-smpl2mhr.yaml` | female-3-casual | SMPL→MHR |

### ZJU MoCap
| Config | Subject | Model |
|--------|---------|-------|
| `zju-377-mhr.yaml` | 377 | MHR |
| `zju-386-mhr.yaml` | 386 | MHR |
| `zju-387-mhr.yaml` | 387 | MHR |
| `zju-392-mhr.yaml` | 392 | MHR |
| `zju-393-mhr.yaml` | 393 | MHR |
| `zju-394-mhr.yaml` | 394 | MHR |
| `zju-377-smplx.yaml` | 377 | SMPL-X |
| `zju-386-smplx.yaml` | 386 | SMPL-X |
| `zju-387-smplx.yaml` | 387 | SMPL-X |
| `zju-392-smplx.yaml` | 392 | SMPL-X |
| `zju-393-smplx.yaml` | 393 | SMPL-X |
| `zju-394-smplx.yaml` | 394 | SMPL-X |

### Evaluation
| Config | Purpose |
|--------|---------|
| `eval-template.yaml` | Template for evaluation runs |
