# Phase 9 — Duplicate-Aware Validation & Generalization Audit

**Environment:** Docker `roundabout-midterm`; Python 3.10.22; TensorFlow 2.13.0; NumPy 1.24.3  
**Scope:** Existing saved models and datasets only. No retraining or input-file/model/submission changes.  
**Interpretation caution:** This repeatedly used validation set is not an independent test set; inferential results below are descriptive and conditional on this dataset.

## Executive Summary

- Original-precision exact cross-split match: **263/5000** validation rows; common non-duplicate subset: **4737** rows.
- Baseline full-validation accuracy: **94.640%**; duplicate subset: **100.000%**; non-duplicate subset: **94.342%**.
- Ensemble full-validation accuracy: **94.960%**; duplicate subset: **100.000%**; non-duplicate subset: **94.680%**.
- On the common subset, Ensemble minus Baseline accuracy: **+0.338 percentage points**; paired bootstrap 95% CI **[-0.253, +0.929]**; exact McNemar p-value **0.28882**.

## A. Exact Duplicate Reproduction

Exact matching uses the source feature values as loaded, without decimal rounding. Keys are tuples of all 25 feature coordinates at their native dtype. Float32 matching is reported separately after casting both datasets.

| Matching representation | Unique train rows | Unique validation rows | Validation rows matching train | Conflicting labels |
|---|---:|---:|---:|---:|
| Native float64 | 33285 | 4822 | 263 | 0 |
| Float32 cast sensitivity | 33285 | 4822 | 263 | 0 |

Within-set rows in excess of unique feature rows: train **1715**, validation **178**. Within-train conflicting feature labels: **0**; within-validation: **0**; cross-split: **0**.

Full list of matching validation row indices (zero-based): `[6, 14, 27, 40, 43, 47, 48, 72, 76, 85, 96, 109, 147, 160, 162, 177, 186, 187, 222, 229, 233, 237, 266, 273, 298, 311, 362, 373, 387, 401, 406, 409, 415, 442, 455, 459, 480, 504, 518, 558, 562, 566, 567, 574, 621, 645, 721, 739, 750, 754, 802, 806, 823, 826, 828, 839, 842, 873, 950, 955, 981, 1002, 1004, 1040, 1046, 1054, 1088, 1094, 1105, 1129, 1156, 1173, 1174, 1193, 1202, 1227, 1278, 1280, 1292, 1299, 1303, 1308, 1327, 1362, 1376, 1381, 1405, 1429, 1440, 1447, 1497, 1498, 1505, 1524, 1525, 1592, 1617, 1620, 1676, 1678, 1683, 1692, 1695, 1718, 1721, 1724, 1743, 1757, 1784, 1789, 1793, 1817, 1846, 1861, 1881, 1902, 1909, 1911, 1935, 1936, 1947, 1994, 2009, 2013, 2020, 2042, 2050, 2149, 2151, 2154, 2183, 2245, 2275, 2278, 2279, 2316, 2321, 2350, 2358, 2414, 2436, 2453, 2456, 2565, 2571, 2582, 2615, 2645, 2700, 2704, 2708, 2800, 2817, 2866, 2867, 2912, 2941, 2979, 2985, 2988, 2999, 3014, 3015, 3024, 3052, 3066, 3113, 3117, 3135, 3137, 3163, 3178, 3198, 3209, 3226, 3240, 3257, 3267, 3292, 3343, 3360, 3367, 3399, 3466, 3469, 3472, 3474, 3476, 3480, 3511, 3544, 3559, 3588, 3596, 3682, 3699, 3706, 3713, 3716, 3722, 3732, 3742, 3787, 3806, 3823, 3853, 3865, 3869, 3881, 3885, 3895, 3916, 3924, 3984, 3986, 4059, 4090, 4112, 4127, 4140, 4144, 4168, 4169, 4203, 4210, 4222, 4302, 4315, 4325, 4333, 4367, 4369, 4397, 4405, 4430, 4451, 4467, 4469, 4511, 4517, 4528, 4538, 4600, 4617, 4629, 4657, 4659, 4660, 4677, 4707, 4769, 4795, 4833, 4854, 4858, 4862, 4878, 4880, 4909, 4926, 4940, 4942, 4949]`.

The JSON contains each matching validation index and all matching training indices. The Float32 sensitivity comparison is not substituted for the native-precision duplicate definition.

## B. Model-specific Training Exposure

The Baseline source training script loads all `train.npz` rows, casting features to Float32 before fitting. Phase 7B saves `split_indices.npz`; the audit independently reconstructed the script's `RandomState(42)` stratified split and compared exact arrays with that saved artifact. See the JSON's `split_reconstruction` for the verification result. Ensemble exposure is the saved 28,000 training indices only; the 7,000 dev rows are excluded.

| Validation exposure definition | Baseline | Ensemble |
|---|---:|---:|
| Matches full train.npz at native precision | 263 | 263 |
| Matches model training inputs at actual Float32 input precision | 263 | 259 |
| Matches actual training inputs but not native full-precision match | 0 | 0 |

For each model, duplicate-aware metrics also include its specific actual-training-input exposure subset. These are exposure flags, not proof that a model memorized a row.

## C. Duplicate-aware Model Evaluation

### Baseline

| Evaluation subset | N | Accuracy | Errors | Macro F1 | Cross entropy |
|---|---:|---:|---:|---:|---:|
| Original validation | 5000 | 94.640% | 268 | 0.9188 | 0.149801 |
| Cross-split exact duplicates | 263 | 100.000% | 0 | 0.4000 | 0.010197 |
| Non-duplicate / common subset | 4737 | 94.342% | 268 | 0.9160 | 0.157551 |
| Matches actual Baseline training inputs | 263 | 100.000% | 0 | 0.4000 | 0.010197 |
| Matches actual Ensemble training inputs | 259 | 100.000% | 0 | 0.4000 | 0.010352 |

Original-validation per-class recall: 0: LANE_LEFT 0.8806; 1: IDLE 0.9449; 2: LANE_RIGHT 0.9250; 3: FASTER 0.9375; 4: SLOWER 0.9578

### Ensemble

| Evaluation subset | N | Accuracy | Errors | Macro F1 | Cross entropy |
|---|---:|---:|---:|---:|---:|
| Original validation | 5000 | 94.960% | 252 | 0.9128 | 0.142650 |
| Cross-split exact duplicates | 263 | 100.000% | 0 | 0.4000 | 0.009715 |
| Non-duplicate / common subset | 4737 | 94.680% | 252 | 0.9105 | 0.150030 |
| Matches actual Baseline training inputs | 263 | 100.000% | 0 | 0.4000 | 0.009715 |
| Matches actual Ensemble training inputs | 259 | 100.000% | 0 | 0.4000 | 0.009864 |

Original-validation per-class recall: 0: LANE_LEFT 0.8657; 1: IDLE 0.9528; 2: LANE_RIGHT 0.8000; 3: FASTER 0.9424; 4: SLOWER 0.9606

Weighted accuracy reconstruction on the original validation set uses the exact duplicate and non-duplicate sample counts and their correct counts. Both reconstructions and the stored-original results are included in the JSON; discrepancies are asserted to be within floating point tolerance.

Macro F1 is the unweighted mean over the five fixed action classes; if a subset has no true or predicted examples for a class, that class contributes F1=0. Per-class recall and support are reported alongside it.

## D. Paired Statistical Comparison on the Common Subset

- Baseline errors corrected by Ensemble: **108**.
- Baseline correct predictions made wrong by Ensemble: **92**.
- Net change: **16** correct predictions; **+0.338 pp**.
- Paired bootstrap percentile 95% CI: **[-0.253, +0.929] pp** using 20,000 paired resamples, seed 20261027.
- Exact two-sided McNemar/binomial p-value: **0.28882** over 200 discordant pairs.

This validation set has been repeatedly used for model selection. These paired tests do not turn it into an independent test set and should not be read as unbiased evidence of future generalization.

## E. Dataset Distribution

For exact duplicates, labels are **1: IDLE: 13, 3: FASTER: 250**. All **263/263** rows have exactly one present vehicle (the first/ego slot under the dataset layout). Ego speed from the first vehicle's vx/vy has mean **0.971**, median **1.000**, and minimum **0.442** in the stored feature scale; the non-duplicate median is **0.533** (mean **0.422**). Thus the exact-duplicate subset is concentrated in higher-speed, ego-only observations, not in stopped/near-zero-speed samples; this is descriptive and does not establish a causal mechanism or a specific speed cap. The JSON reports per-class counts; 25 feature-wise mean, standard deviation, minimum, and maximum; vehicle-presence counts; and each model's confidence and prediction-error rates for duplicate and non-duplicate rows. These are descriptive associations, not causal findings. Vehicle presence is counted from the five presence slots using a >0.5 threshold; ego speed is the Euclidean magnitude of the first vehicle's vx/vy.

## Final Interpretation

### Confirmed Findings

1. Reproduced **263** original-precision exact cross-split duplicate validation rows, with **0** conflicting-label matches.
2. Both models' original, duplicate-only, non-duplicate, and actual-training-exposure metric subsets were evaluated from saved-model probabilities.
3. Baseline duplicate-subset performance implies: **Yes, descriptively: the duplicate subset has higher Baseline accuracy, raising the full-set aggregate by about 0.298 percentage points relative to replacing those rows with non-duplicate performance.**
4. The Ensemble's corresponding descriptive aggregate contribution is **+0.280 pp** (vs. **+0.298 pp** for Baseline); both models score 100% on the 263 duplicates. The overlap impact on the aggregate is similar in size, not evidence of equivalent memorization.
5. On the shared **4737**-sample non-duplicate subset, Ensemble outperforms Baseline by **+0.338 pp**. Its 95% CI includes zero and McNemar p=0.2888; this audit does not provide sufficient evidence by itself to change the submitted model.
6. All **268** Baseline and **252** Ensemble validation errors are on non-duplicate rows. Therefore exact duplicates contribute modestly to the aggregate scores but do not account for the remaining accuracy bottleneck.

### Supported Hypotheses

- If duplicate-subset accuracy exceeds non-duplicate accuracy, exact overlap descriptively contributes to the aggregate score of that model. The same comparison is reported independently for Baseline and Ensemble; similar counts of overlapping rows do not imply equal performance impact.
- The aggregate score uplift is only a counterfactual reweighting description, not a causal estimate of training leakage or memorization.
- Duplicate overlap alone is not sufficient to explain the remaining non-duplicate errors; the measured common-subset error rate quantifies errors outside these exact overlaps.

### Unresolved Limitations

- Validation was used for repeated model selection, so this is not a fresh test-set generalization claim.
- Float32 training-input matching can in principle add matches not present at source precision, but it changed neither unique-row counts nor cross-split matches for these files (263 in both representations).
- Performance differences and distribution summaries are observational; they do not establish that overlap caused a model's accuracy.
- The Ensemble has a small observed common-subset accuracy advantage, but the paired interval crosses zero, the McNemar test is not statistically significant at conventional thresholds, and its macro F1 is lower; this audit alone is not sufficient reason to replace the submission model.

## Artifacts

- Results: `phase9_results.json`
- Figures: `../figures/phase9_*.png`
- Reproduction script: `../scripts/duplicate_validation_audit.py`
