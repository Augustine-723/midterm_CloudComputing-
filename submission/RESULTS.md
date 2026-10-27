# Highway-Env Roundabout Midterm Project: Quantitative Results & Evaluation

> [!IMPORTANT]
> **Evaluation Dataset Clarification:**  
> All accuracy metrics reported below are computed on the course-provided **Validation Set (`dataset/validation.npz`, 5,000 samples)**. They are strictly **Validation Accuracy** and must not be confused with the hidden official Test Set.

---

## 1. Classification Performance Comparison

| Metric | Primary Submission Model (`YOURMODEL.h5` / Baseline) | Custom GradientTape (Single CPU) | Synchronous Distributed (2 Logical CPUs) |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | **94.64%** | 94.12% | **94.60%** |
| **Validation Loss** | 0.1498 | 0.1561 | **0.1493 (Lowest)** |
| **Macro Precision** | **91.03%** | 90.50% | 90.76% |
| **Macro Recall** | **92.92%** | 89.55% | 92.59% |
| **Macro F1-Score** | **91.88%** | 89.93% | 91.64% |
| **Weighted F1-Score** | **94.64%** | 94.11% | **94.60%** |
| **Misclassified Samples** | **268 / 5,000** | 294 / 5,000 | **270 / 5,000** |

---

## 2. Per-Class Recall Breakdown

| Action Class | Sample Count | Baseline (`YOURMODEL.h5`) | Custom GradientTape | Distributed Mirrored |
| :--- | :---: | :---: | :---: | :---: |
| **0: LANE_LEFT** | 134 | 88.06% | **93.28%** | **92.54%** |
| **1: IDLE** | 1,271 | **94.49%** | 94.34% | 92.37% |
| **2: LANE_RIGHT** | 40 | **92.50%** | 72.50% | **87.50%** |
| **3: FASTER** | 1,424 | 93.75% | 91.29% | **93.89%** |
| **4: SLOWER** | 2,131 | 95.78% | 96.34% | **96.67% (Highest)** |

---

## 3. Real Environment Simulation Results (`roundabout-v0`)

Evaluated over 10 rounds $\times$ 10 episodes (100 episodes total) under deterministic paired seeds (`seed_base = 42`):

### 3.1 Ten-Round Integer Rewards (Instructor Format)
- **Primary Model (`YOURMODEL.h5`):**
  ```text
  [88, 97, 92, 83, 91, 94, 95, 90, 81, 98]
  ```
- **Distributed Bonus Model (`distributed_cpu_best.h5`):**
  ```text
  [94, 97, 91, 83, 87, 96, 95, 91, 80, 98]
  ```

### 3.2 Aggregate Environment Metrics

| Environmental Metric | Primary Model (`YOURMODEL.h5`) | Distributed Bonus Model |
| :--- | :---: | :---: |
| **Mean Round Reward (Float Sum)** | **91.50 $\pm$ 5.22** | **91.61 $\pm$ 5.63** |
| **Mean Round Reward (Integer Truncated)** | **90.90 $\pm$ 5.34** | **91.20 $\pm$ 5.79** |
| **Mean Episode Reward** | **9.15 $\pm$ 1.79** | **9.16 $\pm$ 1.81** |
| **Collision Rate** | **14.0%** (14/100) | **14.0%** (14/100) |
| **North Exit Destination Arrival Rate** | **26.0%** (26/100) | **25.0%** (25/100) |
| **Episode Length** | 10.3 steps | 10.3 steps |

> [!NOTE]
> **Destination Arrival Definition:** In `highway-env 1.8.2`, `roundabout-v0`'s `info` dictionary contains no `"arrived"` key. Arrival is evaluated by spatial lane position: reaching the North exit destination lane (`"nxs"`) without collision. Episodes terminate at `time = 11` (11 steps) due to the environment's `duration = 11` configuration.

---

## 4. Controlled Performance Benchmark: Single vs. Distributed

Evaluated on identical hardware (Intel Core i5-9400 CPU @ 2.90GHz, 6 Cores) over 5 timed epochs (1 warm-up epoch excluded):

| Benchmark Metric | Single-Device GradientTape (1 CPU) | Distributed MirroredStrategy (2 Logical CPUs) |
| :--- | :---: | :---: |
| **Replicas in Sync** | 1 | 2 |
| **Avg Time per Epoch** | **0.39 s** | 2.05 s |
| **Total 5-Epoch Duration** | **1.94 s** | 10.23 s |
| **Training Throughput** | **90,360.7 samples/s** | 17,101.1 samples/s |
| **Peak Memory RSS** | **394.8 MB** | 443.3 MB |
| **Relative Speedup** | **1.00x** | **0.19x** |

### Architectural Insight:
On a single physical CPU socket, TensorFlow's single-device C++ Eigen/OpenMP thread pool processes tensors across all 6 cores with zero synchronization overhead. Virtualizing the CPU into 2 Logical Devices with `MirroredStrategy` introduces inter-thread locking and All-Reduce communication overhead per batch, which is mathematically verified and expected on CPU devices.

---

## 5. Known Limitations & Error Analysis
1. **Speed Transition Mutual Confusion:** As discovered in Phase 3B error analysis, 39.55% of all misclassifications occur between `FASTER` (action 3) and `SLOWER` (action 4) during deceleration/acceleration transitions near roundabout entry points.
2. **Dead Observation Feature:** Feature 0 (`Ego_presence`) is identically 1.0 across all 35,000 training and 5,000 validation samples, providing zero discriminative variance.
3. **Speed Boundary Saturation:** In the real simulator, approximately 43% of actions issued near speed boundaries (speed 0 or speed 16) are saturated, though this reflects expert demonstration habits in the dataset rather than model deficiency.

