# Deep Learning Project 1: ResNet on CIFAR-10 under 5M parameters

**Team:** Prachee Gupta, Shini Agarwal  |  **Kaggle team:** Unsupervised

**Final model:** 3-stage ResNet (channels 64/128/256, blocks 3/5/3) with 4,918,602 trainable parameters,
trained on 45,000 images (5,000 held out for validation) with knowledge distillation from the average
of three of our own models, SAM and CutMix. Weights: last epoch (final checkpoint), not a best-validation checkpoint.

| | Accuracy |
|---|---|
| Validation (5,000 held-out CIFAR-10 images, flip TTA with summed logits) | 97.44% |
| Kaggle public leaderboard | 88.8% |
| Kaggle private leaderboard | 85.7% |

The submitted `submission.csv` came from notebook 4 (flip TTA, logits of the image and its mirror image added).

## Notebooks

| Notebook | What it is | Val (flip TTA) | Kaggle public |
|---|---|---|---|
| `1_runs_baseline_A_B.ipynb` | Baseline, Run A, Run B (outputs shown: Run B) | 95.22 / 97.06 / 96.30% | 83.6 / 88.1 / 87.0% |
| `2_runA_full_data.ipynb` | Run A on all 50,000 images | n/a | 88.3% |
| `3_runC_sam_cutmix.ipynb` | Run C: SAM + CutMix, plus test-set analysis | 97.22% | 88.0% |
| `4_final_distillation.ipynb` | **Final model (Run E)**, distilled from A, B and C | **97.44%** | **88.8%** |
| `5_tta_experiment.ipynb` | Test-time augmentation experiment on the final model (probabilities averaged; flip alone gives 97.46%) | | 88.8% |
| `6_inference_only.ipynb` | Makes `submission.csv` from the final weights, no teachers or training | | |

Notebook 1 produced three runs by changing only these settings (outputs saved are from Run B):

| Run | `EPOCHS` | `RUN_NAME` | `augment(x)` returns |
|---|---|---|---|
| Baseline | 30 | `resnet3stage_e30` | `cutout(random_flip(random_crop(x)))` |
| A | 200 | `A_e200` | `cutout(random_flip(random_crop(x)))` |
| B | 200 | `B_strongaug_e200` | `cutout(random_flip(color_jitter(random_zoom(x))))` |

## How to reproduce the final model

Put the Kaggle `data.zip` in your Google Drive (My Drive), open each notebook in Google Colab with an A100 GPU, and run all cells, in this order:

1. Notebook 1 with the **Run A** settings above (saves `A_e200_final_weights.pt`).
2. Notebook 1 again with the **Run B** settings above (saves `B_strongaug_e200_final_weights.pt`).
3. Notebook 3 as it is (Run C, saves `C_sam_cutmix_e300_val_final_weights.pt`).
4. Notebook 4 as it is (final model, saves `E_kd_ensemble_val_final_weights.pt` and writes `submission.csv`).

Running each notebook once with its current defaults does not recreate all three teachers.
Weights and checkpoints are saved to `MyDrive/dl_project1`. Training supports resuming from checkpoints,
but random-number states are not saved, so a resumed run is not bit-for-bit identical.

Training times on an A100: Runs A and B about 15 minutes each, Run C about 50 minutes, final model about 1 hour.

## Inference only

Put `E_kd_ensemble_val_final_weights.pt` in `MyDrive/dl_project1` and run `6_inference_only.ipynb`.
It loads the final student, applies flip TTA (summed logits) and downloads `submission.csv`.Run on a T4 GPU, it reproduces the submitted predictions for 1,999 of the 2,000 test images (the original run used an A100).

No pretrained weights or external data are used. The evaluation images are never used for training.

## References

He et al., Deep Residual Learning (2016); DeVries and Taylor, Cutout (2017); Yun et al., CutMix (2019);
Foret et al., Sharpness-Aware Minimization (2021); Hinton et al., Distilling the Knowledge in a Neural Network (2015).
