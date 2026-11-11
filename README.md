# Deep Learning Project 1: ResNet on CIFAR-10 under 5M parameters

**Team:** Prachee Gupta, Shini Agarwal  |  **Kaggle team:** Unsupervised

**Final model:** 3-stage ResNet with 4,918,602 trainable parameters, trained with
knowledge distillation from the average of three of our own models, SAM and CutMix.

| | Accuracy |
|---|---|
| Validation (5,000 held-out CIFAR-10 images, flip TTA) | 97.44% |
| Kaggle public leaderboard | 88.8% |
| Kaggle private leaderboard | XX.X% |

## Notebooks

| Notebook | What it is | Val (flip TTA) | Kaggle public |
|---|---|---|---|
| `1_runs_baseline_A_B.ipynb` | Baseline, Run A, Run B (outputs shown: Run B) | 95.22 / 97.06 / 96.30% | 83.6 / 88.1 / 87.0% |
| `2_runA_full_data.ipynb` | Run A on all 50,000 images | n/a | 88.3% |
| `3_runC_sam_cutmix.ipynb` | Run C: SAM + CutMix, plus test-set analysis | 97.22% | 88.0% |
| `4_final_distillation.ipynb` | **Final model (Run E)**, distilled from A, B and C | **97.44%** | **88.8%** |
| `5_tta_experiment.ipynb` | Test-time augmentation experiment on the final model | | 88.8% |

Notebook 1 produced three runs by changing only these settings (outputs saved are from Run B):

| Run | `EPOCHS` | `RUN_NAME` | `augment(x)` returns |
|---|---|---|---|
| Baseline | 30 | `resnet3stage_e30` | `cutout(random_flip(random_crop(x)))` |
| A | 200 | `A_e200` | `cutout(random_flip(random_crop(x)))` |
| B | 200 | `B_strongaug_e200` | `cutout(random_flip(color_jitter(random_zoom(x))))` |

## How to reproduce

1. Download the competition data from Kaggle and put `data.zip` in your Google Drive (My Drive).
2. Open a notebook in Google Colab and choose an A100 GPU (Runtime > Change runtime type).
3. Run all cells. Each notebook prints the parameter count and final validation accuracy,
   saves weights to `MyDrive/dl_project1`, and writes `submission.csv`.
4. Run notebook 1 (Runs A and B) and notebook 3 (Run C) before notebook 4,
   because the final model uses their weights as teachers.

Training times on an A100: Runs A and B about 15 minutes each, Run C about 50 minutes,
final model about 1 hour.

No pretrained weights or external data are used. The evaluation images are never used for training.

## References

He et al., Deep Residual Learning (2016); DeVries and Taylor, Cutout (2017); Yun et al., CutMix (2019);
Foret et al., Sharpness-Aware Minimization (2021); Hinton et al., Distilling the Knowledge in a Neural Network (2015).
