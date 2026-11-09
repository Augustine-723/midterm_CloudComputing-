# Phase 10 — Submission Package, Runtime, and Phase 1–9 Audit

## Executive Summary

The refreshed `midterm_roundabout_submission.zip` was extracted into a new directory inside Docker and validated there. The packaged `YOURMODEL.h5` loads with TensorFlow 2.13.0; the packaged `NN_test.py` completed all 100 simulator episodes and returned ten integer round rewards. One-epoch smoke tests of the standard trainer, custom GradientTape trainer, and two-Logical-CPU distributed trainer also exited successfully.

The review found and corrected two packaging weaknesses before rebuilding the ZIP:

1. Three training scripts had a literal `/app/dataset` fallback. Dataset discovery now uses the script directory, its parent, and the current working directory; no computer-specific absolute dataset path remains in the package.
2. The distributed training script previously did not emit the replica/shard/synchronization evidence described by the project report. It now prints replica IDs and per-replica shard sizes, asserts that the shards sum to the global batch, checks all mirrored trainable variables for equality after each epoch, and fails explicitly if two logical CPUs are unavailable.

The package was rebuilt from `submission/`. ZIP CRC validation, fresh extraction, Python syntax compilation, runtime tests, and a text-file scan for machine-specific absolute paths all passed.

## Package Verification

| Check | Result |
|---|---|
| ZIP integrity (`ZipFile.testzip`) | PASS; no bad member |
| Fresh extraction | PASS; 9 files extracted |
| Primary model | `YOURMODEL.h5`, 175,656 bytes |
| Primary model SHA-256 | `ccc4b358d95e9b56805ef8244eb7247d60f9a5d7bc5e5340c197652fef837e02` |
| ZIP SHA-256 | `29abbc0bbbcdf2112b52cc2f906bef583039b7f2caaa674f40516ffb4ab72205` |
| TensorFlow load | PASS, TensorFlow 2.13.0, `compile=False` |
| Model signature | Input `(None, 25)`, output `(None, 5)`, 11,909 parameters |
| Softmax probe | PASS; output rows sum to 1 |
| Packaged `NN_test.py` | PASS; 10 rounds × 10 episodes; exit code 0 |
| Packaged standard `NN_train.py` | PASS; 1-epoch smoke test; loaded 35,000 train and 5,000 validation rows; exit code 0 |
| Packaged `train_gradient_tape.py` | PASS; 1-epoch smoke test; exit code 0 |
| Packaged `train_distributed.py` | PASS; 1-epoch smoke test; exit code 0 |
| Machine-specific path scan | PASS; no `/home`, `/Users`, Windows user path, or `/app` literal in packaged text files |

The `NN_test.py` run returned:

```text
[91, 100, 91, 95, 92, 90, 93, 93, 100, 94]
```

The script does not seed the simulator. These are the observed rewards from this run, not a fixed expected sequence. Each integer is the truncated sum of rewards for one round of ten episodes.

The three one-epoch trainer runs are smoke tests only. Their early-epoch accuracies are not final-model performance claims and were saved to temporary Docker paths, not to the submission package.

## Teacher-Facing Contract Review

The workspace does not contain the instructor's complete grading handout or an authoritative API rubric. The checks below therefore confirm the requirements explicitly documented in the project files and the demonstrated test format; they cannot certify undisclosed grading criteria.

| Deliverable | Observed implementation | Audit |
|---|---|---|
| `NN_train.py` | Loads `data`/`label` from `train.npz` and `validation.npz`; casts to Float32/Int32; constructs 25 → 128 ReLU → 64 ReLU → 5 Softmax; uses Adam, sparse categorical cross-entropy, batch size 64, early stopping/checkpointing; defaults output to `YOURMODEL.h5` beside the script. | PASS against the documented project contract. One-epoch smoke test passed. A full retrain was not run. |
| `NN_test.py` | Loads `YOURMODEL.h5` (or `--model`), imports `highway_env`, constructs `roundabout-v0`, maps a 25-feature observation to `argmax` over five actions, executes 10 rounds × 10 episodes, and prints ten integer reward totals. | PASS against the documented output protocol. Fresh-ZIP full simulator run passed. |
| `YOURMODEL.h5` | HDF5 model with `(None, 25)` input and `(None, 5)` output; SHA matches `models/baseline_standard.h5`. | PASS; no custom objects needed to load. |

Datasets are intentionally excluded from the ZIP. `NN_test.py` can run without them; training requires `train.npz` and `validation.npz` in a `dataset/` folder beside the extracted scripts, in the project root beside `submission/`, or in the current working directory. This is now stated in the packaged README and supported without an absolute path fallback.

### GradientTape Bonus

The packaged custom trainer uses `tf.GradientTape`, explicit gradient calculation and `optimizer.apply_gradients`, a `tf.data.Dataset` pipeline, and a custom validation loop. Static inspection found no calls to `model.fit()`, `model.evaluate()`, or `model.predict()` in that script. A one-epoch smoke test completed successfully. The README no longer claims a separate `test_step` function that is absent from the packaged implementation.

### Synchronous Data Parallelism Bonus

The packaged training implementation configures one physical CPU as two TensorFlow logical CPU devices before runtime initialization, creates `tf.distribute.MirroredStrategy`, distributes a global batch of 64, computes replica-scaled loss, and performs optimizer updates inside `strategy.run()`. The audited one-epoch execution reported:

```text
Logical CPU Devices:   ['/device:CPU:0', '/device:CPU:1']
Replicas in Sync:      2
Global Batch Size:     64 (32 per replica)
[EVIDENCE] Replica IDs: [0, 1]; sharded batch sizes: [32, 32]; global batch: 64
[EVIDENCE] All 6 trainable variables are synchronized across 2 replicas.
```

The implementation now asserts these conditions instead of silently continuing when logical-device setup fails. This is synchronous data parallelism over **two logical devices on one physical CPU**, not two physical processors or two GPUs. Earlier throughput measurements in `reports/phase5b_results.json` show substantial overhead on this CPU setup; no speedup is claimed.

## Phase 1–9 Model Comparison

All classification figures below are validation metrics on the same 5,000-row `dataset/validation.npz`, not independent-test results. Values are transcribed from the saved evaluation reports. Phase 9 independently re-evaluated Baseline and Ensemble predictions from their saved H5 files.

| Phase / model | Training method / exposure | Accuracy | Cross-entropy | Macro F1 | Errors / 5,000 |
|---|---|---:|---:|---:|---:|
| Phase 2 Baseline Standard | Keras `fit`; full 35,000-row training set | 94.64% | 0.1498 | **91.88%** | 268 |
| Phase 3B Wider Network | Wider architecture alternative; full project training experiment | 94.28% | 0.1508 | 90.37% | 286 |
| Phase 3B Lower Learning Rate | Lower learning-rate alternative | 93.84% | 0.1722 | 88.71% | 308 |
| Phase 4 Custom GradientTape | Explicit custom loop; full project training experiment | 94.12% | 0.1561 | 89.93% | 294 |
| Phase 5B Distributed GradientTape | `MirroredStrategy`, 2 logical CPUs | 94.60% | **0.1493** | 91.64% | 270 |
| Phase 7B Seed 42 | Individual model; 28,000 Phase 7B internal-train rows | 94.04% | 0.1688 | 89.89% | 298 |
| Phase 7B Seed 123 | Individual model; 28,000 Phase 7B internal-train rows | 93.76% | 0.1689 | 88.77% | 312 |
| Phase 7B Seed 2026 | Individual model; 28,000 Phase 7B internal-train rows | 94.34% | 0.1579 | 89.77% | 283 |
| Phase 7B Unified Soft-Voting Ensemble | Three seed models, each trained on 28,000 rows | **94.96%** | 0.1426 | 91.28% | **252** |
| Phase 9 Baseline, duplicate-excluded subset | Common 4,737 non-duplicate validation rows | 94.342% | 0.1576 | 91.60% | 268 |
| Phase 9 Ensemble, duplicate-excluded subset | Same common 4,737 rows | 94.680% | 0.1500 | 91.05% | 252 |

The primary packaged `YOURMODEL.h5` remains the Baseline Standard. The Ensemble has the highest observed overall and non-duplicate accuracy, while Baseline has higher overall macro F1. Phase 9's paired comparison on the common subset found 108 Baseline errors corrected by the Ensemble and 92 new Ensemble errors: net +16 correct predictions (+0.338 percentage points), paired bootstrap 95% CI **[−0.253, +0.929] pp**, exact two-sided McNemar p-value **0.289**. This is not sufficient evidence from this repeatedly used validation set alone to justify replacing the packaged model.

## Phase 1–9 Research Conclusions

- **Phase 1:** Confirmed the course dataset and environment were loadable; the dataset is 25-dimensional with five action labels.
- **Phase 2:** Established Baseline Standard as a strong supervised reference, reaching 94.64% validation accuracy.
- **Phase 3A:** Audited the `roundabout-v0` environment and reward behavior. **Phase 3B:** Wider/lower-learning-rate alternatives did not beat Baseline on the saved validation metrics.
- **Phase 4:** Demonstrated custom TensorFlow autodiff using GradientTape; it is a valid implementation alternative, not a measured accuracy improvement over Baseline.
- **Phase 5A–5B:** Audited hardware and implemented synchronous MirroredStrategy on two logical CPUs. Correct replica execution is now instrumented in the packaged training script. This is an architectural demonstration; the single-socket benchmark was slower than single-device training.
- **Phase 6:** Preserved the Baseline as the primary submission model and verified its shape, weights, and simulator-test compatibility in the saved audit.
- **Phase 7B:** The soft-voting ensemble achieved the highest observed validation accuracy (94.96%), but it was built from three models trained on 28,000 internal training rows apiece; it did not dominate Baseline on macro F1.
- **Phase 8:** Found 263 exact train/validation feature overlaps and no exact conflicting labels; also identified local ambiguity in near-neighbor regions and unobserved state variables as plausible error sources.
- **Phase 9:** Reproduced 263 native-Float64 overlaps (also 263 after Float32 casting), exactly reconstructed the Phase 7B training split, and found duplicate-only accuracy of 100% for both models. All 268 Baseline and 252 Ensemble errors were on non-duplicate rows. Overlap modestly lifts aggregate validation accuracy, but does not explain the remaining error ceiling.

## Limitations and Remaining Risks

1. **No independent test set:** The 5,000-row validation set was repeatedly used for model selection and comparison. Metrics and statistical tests are conditional on this dataset and are not fresh-test generalization evidence.
2. **Teacher rubric availability:** The original complete grading rubric is not present in the workspace. Compatibility is confirmed against explicit project documentation and executed behavior, not undisclosed requirements.
3. **Dataset not packaged:** Training scripts require separate dataset placement. The model-loading/environment test does not.
4. **Stochastic environment:** `NN_test.py` does not seed the simulator. Reward totals vary by run; the required output shape and execution were checked, not a fixed reward vector.
5. **Logical CPU semantics:** Two logical CPUs are TensorFlow virtual devices over a single physical CPU. They demonstrate synchronous sharding/replication but do not imply multi-GPU execution or a performance gain.
6. **Model-selection uncertainty:** The Ensemble's common-subset accuracy advantage is small, its paired interval includes zero, and its macro F1 is lower than Baseline. The available evidence does not establish a robust winner on new data.

## Files Reviewed and Produced

- Refreshed package: `midterm_roundabout_submission.zip`
- Package contents and usage: `submission/README.md`
- Packaged result summary: `submission/RESULTS.md`
- Phase 9 duplicate-aware audit: `experiments/phase9/reports/phase9_duplicate_validation.md`
- Phase 9 machine-readable results: `experiments/phase9/reports/phase9_results.json`
- This submission/runtime audit: `reports/phase10_submission_audit.md`
