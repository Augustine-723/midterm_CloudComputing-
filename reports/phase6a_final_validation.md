# Phase 6A — Final Model Integration and Teacher Test Compatibility Report

**Project:** Roundabout Driving Action Prediction (Cloud Computing Midterm)  
**Execution Environment:** Docker Container (`roundabout-midterm`) on Ubuntu Linux  
**Framework:** TensorFlow 2.13.0, Gymnasium 0.29.1, highway-env 1.8.2, Python 3.10.22, NumPy 1.24.3  
**Status:** Completed & 100% Validated (Exit Code 0)  

---

## 1. Executive Summary & Objective

Phase 6A conducts the final pre-submission software engineering audit, model integration, and compatibility testing for the Highway-Env Roundabout Midterm Project.

### Key Milestones Accomplished:
1. **Teacher Codebase Audit**: Thoroughly audited the teacher's original skeleton files (`NN_train.py`, `NN_test.py`, `readme.txt`, `requirements.txt`), identifying key execution mechanics and missing implementation blocks while preserving the original files unmodified.
2. **Standard-Compliant Evaluation Script**: Implemented [`scripts/final_model_test.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/final_model_test.py) adhering strictly to the teacher's 10-round $\times$ 10-episode evaluation protocol without any rule-based intervention, action masking, or environment alterations.
3. **Paired 100-Episode Evaluation**: Evaluated both `models/baseline_standard.h5` and `models/distributed_cpu_best.h5` across 100 deterministic paired episodes in `roundabout-v0`. Both models achieved stable performance with identical 14% collision rates and high mean round rewards (~91.5).
4. **Final Model Recommendation**: Recommended **`models/baseline_standard.h5`** as the primary submission model for maximum Validation Accuracy (94.64%), while retaining **`models/distributed_cpu_best.h5`** as the advanced deliverable for the +10 Data Parallelism bonus.
5. **Automated Submission Integrity Check**: Implemented and executed [`scripts/check_submission.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/check_submission.py), verifying dependencies, dataset integrity, model loading, Softmax normalization, and environment registration with exit code 0.

---

## 2. Teacher Code Audit (`NN_train.py`, `NN_test.py`)

### 2.1 Teacher's Test Workflow Analysis
From inspecting `NN_test.py`:
- **Structure**: 10 rounds executed in `if __name__ == "__main__":`, with each round running 10 episodes (`for _ in range(10):`), yielding 100 total episodes.
- **Per-Round Metric**: Sum of rewards across 10 episodes, returned as an integer: `return int(total_reward)`.
- **Environment Loop**:
  - `(obs, info) = env.reset()`
  - `obs = obs.reshape(1, 25)`
  - Model inference $\to$ `action = np.argmax(...)`
  - `obs, reward, done, truncated, info = env.step(int(action))`
  - Terminate when `done or truncated`.

### 2.2 Missing Code & Critical Nuances in Original `NN_test.py`
| Issue / Missing Block | Observation in Original `NN_test.py` | Resolution in Compatible Script |
| :--- | :--- | :--- |
| **Missing Environment Registration** | Only `import gymnasium as gym` is called. `import highway_env` is missing, causing `gym.make('roundabout-v0')` to throw `NameNotFound`. | Explicitly `import highway_env` to trigger gymnasium registration. |
| **Placeholder Model Loading** | Lines 14–17 contain placeholder comment `==keras.models.load_model==` with no variable assigned. | Implemented CLI argument `--model` to load any `.h5` model dynamically. |
| **Placeholder Action Prediction** | Lines 29–31 contain placeholder comment `==predict(obs)==` with `action` undefined. | Direct forward tensor inference `model(obs_tensor, training=False)` + `np.argmax()`. |
| **Resource Leakage** | `gym.make()` is called in every round, but `env.close()` is never invoked. | Added explicit `env.close()` at termination to free simulator threads and buffers. |
| **Integer Precision Truncation** | `int(total_reward)` truncates decimal reward values. | Tracked both exact floating-point rewards and teacher's integer-truncated format. |

> [!IMPORTANT]
> The original `NN_train.py`, `NN_test.py`, `readme.txt`, and `requirements.txt` provided by the instructor remain **100% unmodified and preserved**.

---

## 3. Model Compatibility & Input/Output Contracts

Both primary models were verified against the strict input/output shape requirements:

| Verification Dimension | Contract Requirement | Baseline Standard (`baseline_standard.h5`) | Distributed CPU Best (`distributed_cpu_best.h5`) | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Input Shape** | `(None, 25)` | `(1, 25)` | `(1, 25)` | **PASSED** |
| **Output Shape** | `(None, 5)` | `(1, 5)` | `(1, 5)` | **PASSED** |
| **Probability Bounds** | Finite, non-negative | All finite $\in [0, 1]$ | All finite $\in [0, 1]$ | **PASSED** |
| **Softmax Normalization** | $\sum_{i=0}^4 P(a_i) = 1.0$ | Sum: `1.0000` | Sum: `1.0000` | **PASSED** |
| **Discrete Action Range** | $\text{argmax} \in \{0, 1, 2, 3, 4\}$ | Valid action (0–4) | Valid action (0–4) | **PASSED** |
| **Direct Tensor Inference** | No `model.predict()` | Direct `model(x)` callable | Direct `model(x)` callable | **PASSED** |

---

## 4. 10 $\times$ 10 Episode Real Environment Evaluation

Both models were evaluated using [`scripts/final_model_test.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/final_model_test.py) on `roundabout-v0` under **identical paired seeds** (`seed_base = 42`).

### 4.1 Ten-Round Total Rewards (Teacher Format: 10 Integer Values)
- **Baseline Standard (`baseline_standard.h5`):**
  ```text
  [88, 97, 92, 83, 91, 94, 95, 90, 81, 98]
  ```
- **Distributed CPU Best (`distributed_cpu_best.h5`):**
  ```text
  [94, 97, 91, 83, 87, 96, 95, 91, 80, 98]
  ```

### 4.2 Quantitative Environment Metrics Comparison

| Evaluation Metric | Baseline Standard (`baseline_standard.h5`) | Distributed CPU Best (`distributed_cpu_best.h5`) | Performance Difference |
| :--- | :---: | :---: | :---: |
| **Mean Round Reward (10 eps)** | **91.50 $\pm$ 5.22** | **91.61 $\pm$ 5.63** | +0.11 in Distributed |
| **Mean Episode Reward** | **9.15 $\pm$ 1.79** | **9.16 $\pm$ 1.81** | +0.01 in Distributed |
| **Mean Episode Duration** | 10.3 steps | 10.3 steps | Identical |
| **Collision Count / Rate** | **14 / 100 (14.0%)** | **14 / 100 (14.0%)** | **Identical Safety Profile** |
| **Arrival / Timeout Rate** | 0% arrived / 86% timeout | 0% arrived / 86% timeout | Identical |
| **Action: 0 (LANE_LEFT)** | 2.61% | 2.42% | -0.19% |
| **Action: 1 (IDLE)** | 23.14% | 24.76% | +1.62% |
| **Action: 2 (LANE_RIGHT)** | 0.97% | 0.87% | -0.10% |
| **Action: 3 (FASTER)** | 25.46% | 25.05% | -0.41% |
| **Action: 4 (SLOWER)** | 47.82% | 46.91% | -0.91% |

#### Behavioral Observation:
Under identical random seeds, both models exhibit near-identical navigation behavior:
- Both models avoid collision in 86 out of 100 episodes.
- Both models prioritize speed regulation (`SLOWER` ~47%, `FASTER` ~25%) while maintaining lane centering (`IDLE` ~24%), closely reflecting the expert driving demonstration distribution.

---

## 5. Final Model Selection & Recommendation

| Selection Criteria | Weight / Priority | Baseline Standard (`baseline_standard.h5`) | Distributed CPU Best (`distributed_cpu_best.h5`) | Superior Option |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Accuracy (Course Rubric)** | **Primary (Highest)** | **94.64%** (268 errors) | 94.60% (270 errors) | **Baseline Standard (+0.04%)** |
| **Validation Loss** | Secondary | 0.1498 | **0.1493** | **Distributed (Lowest)** |
| **Macro F1-Score** | High | **91.88%** | 91.64% | **Baseline Standard (+0.24%)** |
| **Weighted F1-Score** | High | **94.64%** | 94.60% | Baseline Standard |
| **Environment Reward** | Practical | 91.50 $\pm$ 5.22 | **91.61 $\pm$ 5.63** | Distributed (+0.11) |
| **Collision Rate** | Practical | **14.0%** | **14.0%** | Tied |
| **Model Compatibility** | Requirement | 100% Compatible | 100% Compatible | Tied |
| **Bonus Criteria Alignment** | Strategic | Standard Baseline Model | **Advanced Bonus: +10 pts Data Parallelism** | Distributed |

### Definitive Recommendation:
1. **Primary Submission Model (Default Test Candidate):**
   - **`models/baseline_standard.h5`** (Can be mapped or copied to `YOURMODEL.h5`).
   - **Rationale:** The course rubric strictly evaluates model classification accuracy on the hidden test set. `baseline_standard.h5` holds the highest validation accuracy (**94.64%**) and macro F1 (**91.88%**).
2. **Advanced Bonus Deliverable (+10 Points):**
   - **`models/distributed_cpu_best.h5`**
   - **Rationale:** Proves end-to-end multi-replica `tf.distribute.MirroredStrategy` training with `tf.GradientTape`, achieving comparable accuracy (**94.60%**) and the lowest loss (**0.1493**).

---

## 6. Docker Reproducibility Audit (`scripts/check_submission.py`)

Executing `docker exec roundabout-midterm python /app/scripts/check_submission.py`:

```text
======================================================================
 PHASE 6A: SUBMISSION INTEGRITY & REPRODUCIBILITY AUDIT
======================================================================
[1] Python Runtime: 3.10.22

[2] Checking Core Dependencies...
    - NumPy:        1.24.3
    - TensorFlow:   2.13.0
    - Gymnasium:    0.29.1
    - highway-env:  1.8.2

[3] Checking Datasets...
    - train     : data (35000, 25), label (35000, 1) [OK]
    - validation: data (5000, 25), label (5000, 1) [OK]

[4] Checking Trained Models...
    - Baseline Standard   : Loaded successfully | Output Shape: (1, 5) | Softmax Sum: 1.0000 | Action: 1 [OK]
    - Distributed CPU Best: Loaded successfully | Output Shape: (1, 5) | Softmax Sum: 1.0000 | Action: 3 [OK]
    - GradientTape Best   : Loaded successfully | Output Shape: (1, 5) | Softmax Sum: 1.0000 | Action: 1 [OK]

[5] Checking highway-env Environment Registration...
    - roundabout-v0:  Reset and step executed successfully (Obs: (5, 5) -> Next Obs: (5, 5), Action: 1, Reward: 0.917) [OK]

[6] Checking Teacher Original Templates Preservation...
    - NN_train.py    : Present & Preserved [OK]
    - NN_test.py     : Present & Preserved [OK]
    - readme.txt     : Present & Preserved [OK]

======================================================================
 ALL SUBMISSION INTEGRITY & REPRODUCIBILITY AUDITS PASSED [EXIT 0]
======================================================================
```

---

## 7. Submission Readiness Checklist

| Item | Requirement | Verification Result |
| :--- | :--- | :---: |
| **Model Weights** | Available as `.h5` files in `models/` | **READY** |
| **Teacher Code Integrity** | `NN_train.py` & `NN_test.py` untouched | **READY** |
| **Dataset Integrity** | `train.npz` & `validation.npz` intact | **READY** |
| **Reproducibility Script** | `scripts/check_submission.py` exits 0 | **READY** |
| **Evaluation Script** | `scripts/final_model_test.py` exits 0 | **READY** |
| **Advanced Bonus Code** | Phase 4 (`tf.GradientTape`) & Phase 5 (`MirroredStrategy`) | **READY** |

