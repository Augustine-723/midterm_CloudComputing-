# Highway-Env Roundabout-v0 Action Prediction Midterm Project

## 1. Project Objective
This project implements a Deep Neural Network using Supervised Learning to predict discrete driving actions in the `highway-env` (`roundabout-v0`) autonomous driving environment. Given an ego-vehicle's kinematics and surrounding vehicle observations, the model predicts the optimal action from 5 discrete choices:
- `0: LANE_LEFT`
- `1: IDLE`
- `2: LANE_RIGHT`
- `3: FASTER`
- `4: SLOWER`

---

## 2. Development Environment
- **Operating System:** Ubuntu Linux 6.8.0-58-generic (x86_64)
- **Containerization:** Docker (`roundabout-midterm`)
- **Python Version:** 3.10.22
- **Core Dependencies:**
  - `tensorflow==2.13.0`
  - `highway-env==1.8.2`
  - `gymnasium==0.29.1`
  - `numpy==1.24.3`

---

## 3. Dataset Format
- **Training Set (`dataset/train.npz`):** 35,000 samples
- **Validation Set (`dataset/validation.npz`):** 5,000 samples
- **Input Shape:** `(N, 25)` flattened kinematics observations (5 vehicles $\times$ 5 features: presence, x, y, vx, vy)
- **Output Shape:** `(N, 1)` discrete class labels $\in \{0, 1, 2, 3, 4\}$
- **Data Placement:** By default, datasets are expected in `dataset/train.npz` and `dataset/validation.npz` (either in the project root or relative to the script directory).

---

## 4. Neural Network Architecture
The primary model (`YOURMODEL.h5`) utilizes a Multi-Layer Perceptron (MLP) architecture:
- **Input Layer:** `(None, 25)`
- **Dense Layer 1:** 128 units, `ReLU` activation (3,328 params)
- **Dense Layer 2:** 64 units, `ReLU` activation (8,256 params)
- **Output Layer:** 5 units, `Softmax` activation (325 params)
- **Total Trainable Parameters:** 11,909 (46.52 KB)

---

## 5. Model Training (`NN_train.py`)
To train the standard neural network:
```bash
python NN_train.py
```
- **Optimizer:** `Adam(learning_rate=0.001)`
- **Loss:** `SparseCategoricalCrossentropy`
- **Batch Size:** 64
- **Epochs:** Up to 100 with EarlyStopping (`patience=10`)
- **Output:** Saves the optimal checkpoint weights to `YOURMODEL.h5`.

---

## 6. Model Testing (`NN_test.py`)
To evaluate the model in the real `highway-env` `roundabout-v0` simulator:
```bash
python NN_test.py
```
- Executes 10 rounds of 10 episodes (100 episodes total).
- Prints the list of 10 integer total rewards matching the instructor's evaluation format:
  ```text
  [88, 97, 92, 83, 91, 94, 95, 90, 81, 98]
  ```

---

## 7. Advanced Bonus: Custom GradientTape Loop (`train_gradient_tape.py`)
Implements an end-to-end low-level training pipeline using TensorFlow primitives:
- Uses `tf.GradientTape()` context manager for explicit forward pass and gradient calculation.
- Uses `optimizer.apply_gradients()` for explicit weight updates.
- Uses `tf.data.Dataset` pipeline with batching, shuffling, and prefetching.
- **Strictly avoids** `model.fit()`, `model.evaluate()`, and `model.predict()`.
- To run:
  ```bash
  python train_gradient_tape.py
  ```

---

## 8. Advanced Bonus: Synchronous Data Parallelism (`train_distributed.py`)
Implements true multi-device synchronous Data Parallelism via `tf.distribute.MirroredStrategy`:
- **Device Virtualization:** Pre-runtime configuration splits the host physical CPU into 2 virtual Logical CPU devices (`/device:CPU:0`, `/device:CPU:1`).
- **Strategy:** `tf.distribute.MirroredStrategy(devices=['/device:CPU:0', '/device:CPU:1'])`
- **Replicas in Sync:** 2 concurrent replicas.
- **Batch Sharding:** Global Batch Size of 64 is automatically sharded into 32 samples per replica.
- **Loss Scaling & Synchronization:** Uses `tf.nn.compute_average_loss()` and All-Reduce gradient accumulation across replicas via `strategy.run()`.
- **Academic Integrity Note:** Uses two Logical CPU Devices on a single physical CPU socket. No multi-GPU or speedup is claimed for single-socket CPU execution.
- To run:
  ```bash
  python train_distributed.py
  ```

---

## 9. Docker Execution Commands
To run inside the verified Docker container:
```bash
# 1. Start container with project bind-mounted
docker run -it --name roundabout-midterm -v $(pwd):/app -w /app roundabout-midterm bash

# 2. Run test script inside container
docker exec roundabout-midterm python /app/submission/NN_test.py

# 3. Run submission integrity verification
docker exec roundabout-midterm python /app/scripts/verify_final_submission.py
```

---

## 10. Reproduction Checklist
1. Verify `YOURMODEL.h5` exists in the submission directory.
2. Ensure `dataset/train.npz` and `dataset/validation.npz` are present.
3. Run `python NN_test.py` to obtain environment rewards.
4. Run `python NN_train.py` if retraining from scratch is desired.

