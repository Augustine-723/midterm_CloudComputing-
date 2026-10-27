# Phase 5A — Hardware Audit & Feasibility Assessment for TensorFlow Data Parallelism

**Project:** Roundabout Action Prediction (Cloud Computing Midterm)  
**Host System:** Ubuntu Linux 6.8.0-58-generic (x86_64)  
**Execution Environment:** Docker Container (`roundabout-midterm`)  
**TensorFlow Version:** 2.13.0, Python 3.10.22, NumPy 1.24.3  
**Status:** Audit Completed & Verified  

---

## 1. Executive Summary & Objective

In accordance with the course's advanced bonus criteria (**Data Parallelism, up to +10 bonus points**), Phase 5A performs an exhaustive hardware and runtime inspection on both the host machine and the active Docker container.

### Key Audit Findings:
1. **Host Hardware**: The physical host features an **Intel Core i5-9400 CPU** (6 physical cores, 6 logical threads) and a single **NVIDIA GeForce RTX 2060 SUPER** (8 GB VRAM, Driver 580.178.04, CUDA 13.0).
2. **Container Device Access**: The active container `roundabout-midterm` was launched in standard CPU mode without GPU device pass-through (`DeviceRequests: null`, `Devices: []`).
3. **Multi-GPU Feasibility**: Even on the host system, only **one physical GPU** is present. True multi-GPU `MirroredStrategy` requires $\ge 2$ physical GPUs. Under the project guidelines ("Do not modify NVIDIA drivers, do not install CUDA, do not change Docker core dependencies"), Option A (Multi-GPU MirroredStrategy) is physically infeasible.
4. **Feasible & Fully Verified Solution (Option B)**: By leveraging TensorFlow's device virtualization API (`tf.config.set_logical_device_configuration`), the single physical CPU is split into **2 virtual Logical Devices (`/device:CPU:0` and `/device:CPU:1`)** prior to runtime initialization.
5. **Distributed Verification**: `tf.distribute.MirroredStrategy` successfully binds to both logical devices, creating **2 Replicas in Sync**. A minimal synchronous gradient step verified batch partitioning (global batch 8 $\to$ two 4-sample sub-batches) and All-Reduce gradient synchronization.

---

## 2. Host & Container Hardware Inspection

### 2.1 Host Hardware Overview
| Parameter | Value / Specification | Command Used |
| :--- | :--- | :--- |
| **CPU Model** | Intel(R) Core(TM) i5-9400 @ 2.90GHz (Max 4.10GHz) | `lscpu` |
| **Physical Sockets** | 1 Socket | `lscpu` |
| **Physical Cores** | 6 Cores (6 cores per socket) | `lscpu` |
| **Logical Threads** | 6 Threads (1 thread per core, SMT/HyperThreading disabled) | `lscpu` |
| **CPU Instruction Sets** | AVX2, FMA, SSE4_2, AES | `lscpu` |
| **Host System Memory** | Total: 31.27 GB \| Available: 8.98 GB | `free -h` / `/proc/meminfo` |
| **Host GPU Model** | 1x NVIDIA GeForce RTX 2060 SUPER (8,192 MiB) | `nvidia-smi` |
| **NVIDIA Driver** | 580.178.04 | `nvidia-smi` |
| **Host CUDA Toolkit** | 13.0 | `nvidia-smi` |

### 2.2 Docker Device Availability
- **Docker Runtimes:** `runc`, `io.containerd.runc.v2`
- **Container Name:** `roundabout-midterm` (Container ID: `389b5d110c2b`)
- **GPU Pass-Through (`docker inspect`):**
  - `Devices`: `[]` (None passed)
  - `DeviceRequests`: `null` (No `--gpus` flag specified at launch)
- **Container Access Status:** Pure CPU container. `nvidia-smi` and CUDA driver stubs are not bound inside the container.

---

## 3. TensorFlow Device Inspection

Inside the Docker container:

```python
import tensorflow as tf

physical_cpus = tf.config.list_physical_devices('CPU')
physical_gpus = tf.config.list_physical_devices('GPU')

# Output:
# physical_cpus: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
# physical_gpus: []
```

- **Physical CPU:** 1 device (`/physical_device:CPU:0`)
- **Physical GPU:** 0 devices

---

## 4. Distribution Strategy Feasibility Assessment

### Comparative Evaluation of Options

| Strategy Option | Hardware Requirements | Current Availability | Feasibility Assessment |
| :--- | :--- | :---: | :--- |
| **Option A: Multi-GPU MirroredStrategy** | $\ge 2$ physical NVIDIA GPUs with NVLink / PCIe | **0 GPUs in container** (1 on host) | **Infeasible** (Host has only 1 GPU; multi-GPU impossible without physical hardware). |
| **Option B: CPU Logical Devices MirroredStrategy** | Multi-core CPU with $\ge 2$ logical device configurations | **6 physical CPU cores available** | **100% Feasible & Verified** (TensorFlow officially supports logical devices for distributed simulation). |
| **Option C: Infeasible / Abort** | Single-core CPU without virtualization | N/A | **Rejected** (Option B works robustly). |

> [!IMPORTANT]
> **Definitive Recommendation: Option B (Multi-Logical-Device CPU MirroredStrategy)**.  
> This allows the complete TensorFlow distributed architecture—including `MirroredStrategy`, `experimental_distribute_dataset`, distributed `tf.GradientTape`, per-replica loss computation, and `strategy.reduce` gradient synchronization—to be implemented and demonstrated truthfully without misrepresenting hardware capabilities.

---

## 5. Conceptual Distinctions (Academic Rigor)

To strictly adhere to engineering honesty and academic integrity, the following conceptual boundaries are maintained:

```
Physical Hardware Level:
+-----------------------------------------------------------+
| 1 Physical CPU Socket (Intel i5-9400, 6 Cores, 6 Threads) |
+-----------------------------------------------------------+
                             |
TensorFlow Runtime Level:    v (tf.config.set_logical_device_configuration)
+-------------------------------+   +-------------------------------+
| Logical Device: /device:CPU:0 |   | Logical Device: /device:CPU:1 |
+-------------------------------+   +-------------------------------+
                             |                               |
tf.distribute.MirroredStrategy: v                               v
+-------------------------------+   +-------------------------------+
|  Replica 0 (Worker 0)         |   |  Replica 1 (Worker 1)         |
|  - Graph Copy                 |   |  - Graph Copy                 |
|  - Sub-batch: [32, 25]        |   |  - Sub-batch: [32, 25]        |
|  - Local Loss & Gradient      |   |  - Local Loss & Gradient      |
+-------------------------------+   +-------------------------------+
                             \                               /
                              \==== ALL-REDUCE (SUM) =======/
                                             |
                                Synchronized Global Weight Update
```

1. **Physical Device vs. Logical Device**:
   - A **Physical Device** represents actual hardware registered by the operating system (e.g., 1 physical CPU package).
   - A **Logical Device** is an isolated virtual computing device instantiated inside TensorFlow's runtime memory space. Splitting 1 physical CPU into 2 logical CPUs creates 2 distinct device contexts (`/device:CPU:0`, `/device:CPU:1`). **We explicitly DO NOT claim that two logical CPUs are two physical CPUs or two GPUs.**
2. **Replica**:
   - A **Replica** is an execution instance assigned to a specific device. Each replica maintains a replicated copy of the model variables and executes forward/backward passes on a slice of the global batch.
3. **Data Parallelism vs. Model Parallelism**:
   - In **Data Parallelism**, the model is identical across all replicas, but the dataset is partitioned (sharded). Replicas compute local gradients independently on disjoint data subsets, then synchronize gradients via **All-Reduce** before updating weights.
4. **Multi-GPU vs. Multi-Logical CPU Data Parallelism**:
   - The algorithmic graph and synchronization protocols (`MirroredStrategy`, `NCCL` or `Ring All-Reduce`, `strategy.run`, `strategy.reduce`) are **identical**. The only distinction is whether tensor kernels execute on CUDA stream units or host thread pools.

---

## 6. Minimal Distributed Test Execution & Evidence

To prove functionality, [`scripts/check_distributed_devices.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/check_distributed_devices.py) was executed in Docker.

### 6.1 Setup Code
```python
# Must be executed BEFORE any TensorFlow runtime tensors are allocated
cpus = tf.config.list_physical_devices('CPU')
tf.config.set_logical_device_configuration(
    cpus[0],
    [
        tf.config.LogicalDeviceConfiguration(),
        tf.config.LogicalDeviceConfiguration()
    ]
)

strategy = tf.distribute.MirroredStrategy(devices=['/device:CPU:0', '/device:CPU:1'])
```

### 6.2 Execution Output
```text
[3] Configuring Logical CPU Devices:
    -> Successfully configured physical CPU:0 into 2 virtual Logical Devices.
    - Logical CPUs in Runtime: 2 -> ['/device:CPU:0', '/device:CPU:1']
    - Logical GPUs in Runtime: 0 -> []

[4] Initializing tf.distribute.MirroredStrategy:
    - Devices assigned: ['/device:CPU:0', '/device:CPU:1']
    - Replicas in sync: 2

[5] Executing Minimal Synchronous Distributed Training Step:
    - Distributed Step Succeeded!
    - Total Scaled Loss: 1.6741
    - Replica IDs in Sync: (<tf.Tensor: shape=(), dtype=int32, numpy=0>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
    - Sharded Batch Sizes: (<tf.Tensor: shape=(), dtype=int32, numpy=4>, <tf.Tensor: shape=(), dtype=int32, numpy=4>)
```

### 6.3 Empirical Proof:
- **Replica IDs Verified:** Replica 0 and Replica 1 both actively executed the computation graph.
- **Batch Sharding Verified:** The global batch of size 8 was automatically partitioned into **two distinct batches of size 4**.
- **Gradient Synchronization Verified:** Local gradients from both replicas were combined via `strategy.reduce(tf.distribute.ReduceOp.SUM)` and applied to the shared model variables.

---

## 7. Phase 5B Training Design

For Phase 5B (Full Data Parallelism Training), the architecture will seamlessly extend Phase 4's custom `tf.GradientTape` workflow into a distributed paradigm:

### 1. Batch Sharding
- **Global Batch Size:** 64
- **Per-Replica Batch Size:** 32 (automatically handled by `strategy.experimental_distribute_dataset(dataset)`)
- In `roundabout-v0`, 35,000 training samples $\to$ $\lceil 35,000 / 64 \rceil = 547$ distributed steps per epoch.

### 2. Per-Replica Loss Computation
- Cross-entropy loss must be configured with `reduction=tf.keras.losses.Reduction.NONE`.
- Inside each replica, loss is scaled by global batch size:
  $$\mathcal{L}_{\text{replica}} = \text{tf.nn.compute\_average\_loss}(\text{per\_example\_loss}, \text{global\_batch\_size}=64)$$

### 3. Gradient Synchronization
- Each replica computes local gradients $\nabla_{\theta} \mathcal{L}_i$ via `tape.gradient()`.
- Across replicas, gradients are summed via All-Reduce:
  $$\nabla_{\theta} \mathcal{L}_{\text{global}} = \text{strategy.reduce}(\text{ReduceOp.SUM}, \nabla_{\theta} \mathcal{L}_{\text{replica}}, \text{axis}=\text{None})$$

### 4. Synchronous Model Weight Update
- Updated weights are computed synchronously:
  $$\theta \leftarrow \theta - \eta \cdot \nabla_{\theta} \mathcal{L}_{\text{global}}$$
- Executed via `optimizer.apply_gradients(zip(reduced_grads, model.trainable_variables))`.

### 5. Verification & Telemetry
- Validate accuracy and loss against `dataset/validation.npz`.
- Log `strategy.experimental_local_results` at Epoch 1 to provide concrete evidence of multi-replica execution in training logs.

---

## 8. Deliverables & Audit Summary

1. **Audit Script**: [`scripts/check_distributed_devices.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/check_distributed_devices.py)
2. **Audit Telemetry JSON**: [`reports/phase5a_hardware_audit.json`](file:///home/augustine/cloud_computing/midterm_roundabout/reports/phase5a_hardware_audit.json)
3. **Audit Markdown Report**: [`reports/phase5a_hardware_audit.md`](file:///home/augustine/cloud_computing/midterm_roundabout/reports/phase5a_hardware_audit.md)

