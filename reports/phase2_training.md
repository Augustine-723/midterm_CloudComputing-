# Phase 2 成果報告：Baseline Neural Network 訓練與評估

> **專案名稱**：雲端計算期中專題 — Supervised Learning for Highway-env Roundabout  
> **專案路徑**：`/home/augustine/cloud_computing/midterm_roundabout`  
> **執行環境**：Docker Container (`roundabout-midterm`) on Ubuntu  
> **Python / TensorFlow / NumPy**：`3.10.22` / `2.13.0` / `1.24.3`  
> **模型格式**：Keras HDF5 (`.h5`)  
> **訓練腳本**：`scripts/train_baseline.py`  
> **評估腳本**：`scripts/evaluate_model.py`  
> **完成日期**：2026-10-22

---

## 1. Dataset Summary（資料集摘要）

本階段訓練使用 Phase 1 嚴格校驗之官方資料集，資料來源完全隔離，絕無跨集污染：

- **訓練集 (`dataset/train.npz`)**：
  - 特徵維度：`(35000, 25)`，數值已在 $[-1.0, 1.0]$ 區間（Kinematics 歸一化），轉為 `float32`，未施加破壞性二次標準化。
  - 標籤維度：`(35000,)`，轉為 `int32`，數值為 $0 \sim 4$ 五個離散動作。
- **驗證集 (`dataset/validation.npz`)**：
  - 特徵維度：`(5000, 25)`，`float32`。
  - 標籤維度：`(5000,)`，`int32`。
- **資料分佈與類別統計**：
  - `0: LANE_LEFT`：訓練集 806 筆 (**2.30%**)，驗證集 134 筆 (**2.68%**)
  - `1: IDLE`：訓練集 8,710 筆 (**24.89%**)，驗證集 1,271 筆 (**25.42%**)
  - `2: LANE_RIGHT`：訓練集 305 筆 (<font color="red">**0.87%**</font>)，驗證集 40 筆 (<font color="red">**0.80%**</font>)
  - `3: FASTER`：訓練集 9,986 筆 (**28.53%**)，驗證集 1,424 筆 (**28.48%**)
  - `4: SLOWER`：訓練集 15,193 筆 (**43.41%**)，驗證集 2,131 筆 (**42.62%**)

---

## 2. Model Architecture（模型架構）

依專案規範，使用 `tf.keras.Sequential` 建立標準前饋神經網路（Feedforward Neural Network）：

```text
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_state (InputLayer)    (None, 25)                0         
 dense_1 (Dense)             (None, 128)               3328      
 dense_2 (Dense)             (None, 64)                8256      
 output_action (Dense)       (None, 5)                 325       
=================================================================
Total params: 11,909 (46.52 KB)
Trainable params: 11,909 (46.52 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

- **輸入層**：25 維狀態向量（包含圓環周圍 5 輛車之 presence, x, y, vx, vy）。
- **隱藏層 1**：128 神經元，ReLU 激活函數。
- **隱藏層 2**：64 神經元，ReLU 激活函數。
- **輸出層**：5 神經元，Softmax 激活函數，代表 5 個離散動作之預測機率分佈。

---

## 3. Hyperparameters & Training Settings（超參數配置）

- **隨機種子 (Random Seed)**：Python、NumPy、TensorFlow 皆統一固定為 `42`，確保完全可重現。
- **優化器 (Optimizer)**：Adam，學習率 $\eta = 0.001$。
- **損失函數 (Loss)**：`SparseCategoricalCrossentropy()`。
- **評估指標 (Metric)**：`SparseCategoricalAccuracy()`。
- **批次大小 (Batch Size)**：`64`（每 Epoch 547 個 Iterations）。
- **最大訓練輪數 (Max Epochs)**：`100`。
- **回呼機制 (Callbacks)**：
  - `EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)`
  - `ModelCheckpoint(monitor="val_loss", save_best_only=True)`
- **模型儲存格式**：以標準 HDF5 (`.h5`) 格式持久化儲存。

---

## 4. Experiment A Results（標準基準成果）

- **設定**：無類別加權（Unweighted）。
- **儲存路徑**：`models/baseline_standard.h5`
- **訓練耗時**：31.14 秒。
- **最佳 Epoch**：第 **46** 輪（於第 56 輪觸發 Early Stopping 並自動還原權重）。
- **Validation Loss**：`0.1498`
- **Validation Accuracy**：**94.64%**
- **Macro Precision / Recall / F1**：91.03% / 92.92% / **91.88%**
- **Weighted F1-score**：**94.64%**

**各類別詳細表現**：
| 類別代碼與動作 | Support | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0: LANE_LEFT` | 134 | 118 | 12 | 16 | 90.77% | **88.06%** | 89.39% |
| `1: IDLE` | 1,271 | 1,201 | 73 | 70 | 94.27% | 94.49% | 94.38% |
| `2: LANE_RIGHT` | 40 | 37 | 9 | 3 | 80.43% | **92.50%** | 86.05% |
| `3: FASTER` | 1,424 | 1,335 | 93 | 89 | 93.49% | 93.75% | 93.62% |
| `4: SLOWER` | 2,131 | 2,041 | 81 | 90 | 96.18% | 95.78% | 95.98% |

---

## 5. Experiment B Results（類別加權基準成果）

- **設定**：依訓練集分佈計算 Balanced Class Weights，並於 `model.fit(..., class_weight=weights)` 導入。
  $$w_i = \frac{N}{K \cdot N_i}$$
  - $w_0$ (`LANE_LEFT`): **`8.6849`**
  - $w_1$ (`IDLE`): **`0.8037`**
  - $w_2$ (`LANE_RIGHT`): **`22.9508`**
  - $w_3$ (`FASTER`): **`0.7010`**
  - $w_4$ (`SLOWER`): **`0.4607`**
- **儲存路徑**：`models/baseline_weighted.h5`
- **訓練耗時**：32.01 秒。
- **最佳 Epoch**：第 **42** 輪（於第 52 輪觸發 Early Stopping 並自動還原權重）。
- **Validation Loss**：`0.2055`
- **Validation Accuracy**：**92.46%**
- **Macro Precision / Recall / F1**：82.16% / 92.97% / **86.44%**
- **Weighted F1-score**：**92.55%**

**各類別詳細表現**：
| 類別代碼與動作 | Support | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0: LANE_LEFT` | 134 | 129 | 48 | 5 | 72.88% | <font color="green">**96.27% (+8.2%)**</font> | 82.96% |
| `1: IDLE` | 1,271 | 1,156 | 102 | 115 | 91.89% | 90.95% | 91.42% |
| `2: LANE_RIGHT` | 40 | 37 | 27 | 3 | 57.81% | **92.50%** | 71.15% |
| `3: FASTER` | 1,424 | 1,298 | 94 | 126 | 93.25% | 91.15% | 92.19% |
| `4: SLOWER` | 2,131 | 2,003 | 106 | 128 | 94.97% | 93.99% | 94.48% |

---

## 6. Comparison Table（綜合評比對照表）

| 評估指標 | Exp A: Standard (未加權) | Exp B: Class Weighted (加權) | 差異比較 |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy (主要評分)** | <font color="green">**94.64%**</font> | 92.46% | Exp A 高出 **+2.18%** |
| **Validation Loss** | <font color="green">**0.1498**</font> | 0.2055 | Exp A 損失較低（預測校準更佳） |
| **Macro Precision** | <font color="green">**91.03%**</font> | 82.16% | Exp A 誤報（FP）顯著較少 |
| **Macro Recall** | 92.92% | <font color="green">**92.97%**</font> | Exp B 略微領先 |
| **Macro F1-score** | <font color="green">**91.88%**</font> | 86.44% | Exp A 綜合平衡度高 |
| **Weighted F1-score** | <font color="green">**94.64%**</font> | 92.55% | Exp A 總體加權 F1 高出 **+2.09%** |
| **Class 0 (LANE_LEFT) Recall** | 88.06% (118/134) | <font color="green">**96.27% (129/134)**</font> | **Exp B 大幅提升 +8.21%** |
| **Class 2 (LANE_RIGHT) Recall** | **92.50% (37/40)** | **92.50% (37/40)** | 兩者持平，僅漏抓 3 筆 |
| **Class 2 (LANE_RIGHT) Precision**| <font color="green">**80.43%**</font> (FP=9) | 57.81% (FP=27) | Exp B 權重過重導致假陽性激增 |

---

## 7. Confusion Matrix Analysis（混淆矩陣深入剖析）

### 7.1 混淆矩陣對照

**Exp A (Standard Baseline)**：
```text
           Pred 0   Pred 1   Pred 2   Pred 3   Pred 4
True 0 [      118       10        0        3        3 ] (Recall: 88.1%)
True 1 [        5     1201        5       35       25 ] (Recall: 94.5%)
True 2 [        0        1       37        2        0 ] (Recall: 92.5%)
True 3 [        4       29        3     1335       53 ] (Recall: 93.8%)
True 4 [        3       33        1       53     2041 ] (Recall: 95.8%)
```

**Exp B (Class Weighted Baseline)**：
```text
           Pred 0   Pred 1   Pred 2   Pred 3   Pred 4
True 0 [      129        5        0        0        0 ] (Recall: 96.3%)
True 1 [       29     1156       13       35       38 ] (Recall: 91.0%)
True 2 [        0        1       37        2        0 ] (Recall: 92.5%)
True 3 [        8       45        5     1298       68 ] (Recall: 91.2%)
True 4 [       11       51        9       57     2003 ] (Recall: 94.0%)
```

### 7.2 剖析與權衡 (Trade-off)
1. **少數類別捕捉力**：
   - 在未加權的 Exp A 中，Class 0 的漏抓（FN）有 16 筆（主要被誤判為 IDLE 10 筆）；而在 Exp B 中，由於給予了 8.68 倍損失權重，Class 0 的召回率大幅拉升至 **96.27%**，僅漏抓 5 筆，且完全沒有被誤判為加速或減速。
2. **假陽性激增（False Positives 暴增）代價**：
   - 由於 Class 2 權重高達 **22.95**，Exp B 的神經網路傾向於在稍微有右轉跡象時便大膽預測 Class 2，導致預測 Class 2 的誤報數從 9 次暴增到 27 次，使得 Class 2 的 Precision 從 80.43% 跌落至 57.81%。
   - 同樣地，Class 0 的預測誤報亦從 12 次增加至 48 次，主要來自直行 (IDLE, 29 次) 與減速 (SLOWER, 11 次) 的侵蝕。
3. **Accuracy 下降之本質**：
   - Exp B 犧牲了多數類別（IDLE、FASTER、SLOWER）的極致準確性，來換取少數類別的高敏銳度。這直接反映在整體 Validation Accuracy 上（94.64% $\rightarrow$ 92.46%）。

---

## 8. Best Model Selection（最佳模型決策建議）

在目前兩版 Baseline 之間，存在明確的指標取捨（Trade-off）：

1. **若以「教師成績評分標準（Validation Accuracy）」為第一考量**：
   - **推薦選擇 `baseline_standard.h5` (Exp A)**。
   - 原因：具備高達 **94.64%** 的 Validation Accuracy，Macro Precision 高達 91.03%，且在只有 40 筆的極少數類別 Class 2 上，其 Recall 依然達到 92.50%，綜合表現穩定。
2. **若以「實際圓環自駕避障與變道成功率」為第一考量**：
   - **`baseline_weighted.h5` (Exp B)** 在 Class 0 具備極強的主動出彎/左變道意圖（Recall 96.27%）。但需注意過度積極的變道決策（高 FP）可能在特定動態路況下造成非預期的變道抖動。

> **最終決策**：
> 目前先將 **`baseline_standard.h5`** 定位為專題的主要基準模型（符合老師要求之高準確率與平衡表現）；同時保留 **`baseline_weighted.h5`** 作為在 highway-env 實車模擬測試時對比行車動態之備選方案。

---

## 9. Identified Problems（發現之潛在問題）

1. **極端權重造成的過度反應 (Over-sensitivity)**：
   - 直接採用反比例權重使 Class 2 權重飆升到 22.95，幅度過大。未來不宜使用全額均衡加權，而應採用溫和加權（例如取平方根權重 $\sqrt{w_i}$ 或 Focal Loss $\gamma=2.0$）。
2. **特徵交互表徵能力受限 (Feature Representation)**：
   - 目前模型僅使用 2 層純 Dense 網路處理展平後的 25 維特徵。然而，輸入資料本質上是由 5 輛車各 5 個時空屬性構成（主車 + 4 輛周遭車輛）。純 MLP 缺乏對不同車輛之間相對幾何與交互關係的歸納偏置（Inductive Bias）。
3. **過擬合起點 (Overfitting Point)**：
   - 觀察 Loss 曲線，兩模型在第 30~45 Epoch 左右均進入驗證損失瓶頸區，後續雖然訓練損失繼續下降，但驗證損失不再改善，顯示適度的 Dropout 或 BatchNormalization 將有助於穩定收斂。

---

## 10. Phase 3 Recommendations（Phase 3 演進建議）

若後續獲得核准進入 Phase 3，建議以下具體優化方向：

1. **改良損失函數 (Focal Loss / Soft Weighting)**：
   - 導入 Multi-class Focal Loss：針對易分類樣本（多數類別）自動降權，針對困難樣本（少數類別）動態聚焦，兼顧 94%+ 高準確率與 95%+ 高 Recall，避免 Class Weight 帶來的 Precision 斷崖。
2. **架構進化 (Structured MLP / Residual Connections)**：
   - 加入 `BatchNormalization` 與輕量 `Dropout(0.2)`，降低對特定噪聲的敏感度。
   - 探討對 5 輛車進行局部特徵編碼（如 1D-Conv 或 Dense 特徵映射後再聚合），保留自駕場景的車輛結構資訊。
3. **環境閉環模擬測試 (Closed-loop Testing in `roundabout-v0`)**：
   - 製作不修改原始 [`NN_test.py`](file:///home/augustine/cloud_computing/midterm_roundabout/NN_test.py) 的評估腳本，直接將兩版模型載入 `roundabout-v0` 運行 100 輪 Episode，計算真實平均回報 (Average Reward) 與碰撞率 (Crash Rate)，建立離線指標與在線性能的關聯。

---

## 11. 重現性與模型驗證記錄 (Reproducibility & Model Verification)

在 Docker 容器內以 `tf.keras.models.load_model()` 重新載入，確認推論相容性：
- `models/baseline_standard.h5`：輸入 `shape=(1, 25)` $\rightarrow$ 輸出 `shape=(1, 5)`，機率和為 1.0，推論測試 **PASS**。
- `models/baseline_weighted.h5`：輸入 `shape=(1, 25)` $\rightarrow$ 輸出 `shape=(1, 5)`，機率和為 1.0，推論測試 **PASS**。

