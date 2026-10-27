# Phase 1 驗證報告：資料集與環境相容性診斷

> **專案名稱**：雲端計算期中專題 — Supervised Learning for Highway-env Roundabout  
> **專案路徑**：`/home/augustine/cloud_computing/midterm_roundabout`  
> **執行環境**：Docker Container (`roundabout-midterm`) on Ubuntu  
> **診斷日期**：2026-10-22  
> **診斷工具**：`scripts/check_roundabout.py`、`scripts/inspect_dataset.py`

---

## 1. Task A — 專案檔案清單檢查 (Project Files Inspection)

檢查專案目錄 `/home/augustine/cloud_computing/midterm_roundabout` 之關鍵檔案：

| 檔案名稱 | 存在狀態 | 說明 |
| :--- | :---: | :--- |
| `Dockerfile` | **存在** | 已配置 Python 3.10-slim，鎖定 `gymnasium==0.29.1` |
| `requirements.txt` | **存在** | 包含 highway-env, tensorflow==2.13.0 |
| `NN_train.py` | **存在** | 老師提供之訓練骨架，**完全未修改** |
| `NN_test.py` | **存在** | 老師提供之測試評估骨架，**完全未修改** |
| `dataset/train.npz` | **存在** | 包含 `data` (35000, 25) 與 `label` (35000, 1) |
| `dataset/validation.npz` | **存在** | 包含 `data` (5000, 25) 與 `label` (5000, 1) |
| `dataset/train/` | **存在** | 原始上傳之 `data.npy` 與 `label.npy` |
| `dataset/validation/` | **存在** | 原始上傳之 `data.npy` 與 `label.npy` |

專案關鍵檔案與資料集皆已就位，老師提供之原始檔案皆完整保留。

---

## 2. Task B — 資料集統計與檢驗 (Dataset Validation)

透過專用診斷程式 [`scripts/inspect_dataset.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/inspect_dataset.py) 於 Docker 容器內讀取並檢驗，12 項檢查結果如下：

### 2.1 訓練集與驗證集綜合統計表

| 檢驗項目 | 訓練集 (`train.npz`) | 驗證集 (`validation.npz`) | 一致性與相容判定 |
| :--- | :--- | :--- | :---: |
| **1. Dataset Keys** | `['data', 'label']` | `['data', 'label']` | **完全一致** |
| **2. Feature Shape** | `(35000, 25)` | `(5000, 25)` | **完全一致** (皆為 25 維) |
| **3. Label Shape** | `(35000, 1)` | `(5000, 1)` | **完全一致** (皆為單一動作欄位) |
| **4. Feature Dtype** | `float64` | `float64` | **完全一致** (建議轉 `float32`) |
| **4. Label Dtype** | `float64` | `float64` | **完全一致** (需轉為 `int`) |
| **5. 樣本數 (Samples)** | **35,000** 筆 | **5,000** 筆 | 比例約為 $7:1$ |
| **6. 單一樣本特徵數** | **25** (或 $5 \times 5$) | **25** (或 $5 \times 5$) | **符合規格** |
| **7. 獨立標籤 (Unique)**| `[0.0, 1.0, 2.0, 3.0, 4.0]` | `[0.0, 1.0, 2.0, 3.0, 4.0]` | **完全落在 0~4 類別** |
| **9. 缺失值 (NaN)** | **0** | **0** | **無缺失值** |
| **10. 無限值 (Inf)** | **0** | **0** | **無極端異常值** |
| **11. 特徵最小值 (Min)**| `-1.000000` | `-1.000000` | **已歸一化** |
| **11. 特徵最大值 (Max)**| `1.000000` | `1.000000` | **已歸一化** |
| **11. 特徵均值 (Mean)** | `0.172630` | `0.172021` | 分佈高度一致 |
| **11. 特徵標準差 (Std)**| `0.553126` | `0.552532` | 分佈高度一致 |

### 2.2 類別分佈 (Label Distribution)

| 動作代碼 | 動作意義 (highway-env) | 訓練集樣本數 | 訓練集佔比 | 驗證集樣本數 | 驗證集佔比 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | `LANE_LEFT` (左換道) | 806 | **2.30%** | 134 | **2.68%** |
| **1** | `IDLE` (保持航向) | 8,710 | **24.89%** | 1,271 | **25.42%** |
| **2** | `LANE_RIGHT` (右換道) | 305 | <font color="red">**0.87%**</font> | 40 | <font color="red">**0.80%**</font> |
| **3** | `FASTER` (加速) | 9,986 | **28.53%** | 1,424 | **28.48%** |
| **4** | `SLOWER` (減速) | 15,193 | **43.41%** | 2,131 | **42.62%** |
| **總計**| | **35,000** | **100.00%** | **5,000** | **100.00%** |

> [!IMPORTANT]
> **類別嚴重不平衡（Severe Class Imbalance）警示**：
> - 動作 4 (減速)、動作 3 (加速)、動作 1 (保持) 合計佔全體資料集的 **96.8%**。
> - 動作 0 (左轉/左換道) 僅佔 **2.3%**，動作 2 (右轉/右換道) 更僅佔 **0.87%**。
> - 若模型以傳統損失函數進行單純訓練，模型容易陷入「永不換道」的局部最佳解，導致在圓環測試中無法順利出入匝道。在 Phase 2 必須引入**類別權重 (Class Weights)** 或特定損失函數以確保換道行為的學習。

---

## 3. Task C — 環境相容性診斷 (Environment Compatibility)

透過 [`scripts/check_roundabout.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/check_roundabout.py) 於 Docker 容器內實測 `roundabout-v0`：

### 3.1 套件版本實測
- **Python**：`3.10.22`
- **TensorFlow**：`2.13.0`
- **NumPy**：`1.24.3`
- **highway-env**：`1.8.2`
- **Gymnasium**：`0.29.1`

### 3.2 `roundabout-v0` 環境規格
1. **Observation Space**：`Box(-inf, inf, (5, 5), float32)`，展平後為 **25** 維。
2. **Action Space**：`Discrete(5)`，動作集合為 $\{0, 1, 2, 3, 4\}$。
3. **Observation Configuration**：
   ```json
   {
       "type": "Kinematics",
       "absolute": true,
       "features_range": {
           "x": [-100, 100],
           "y": [-100, 100],
           "vx": [-15, 15],
           "vy": [-15, 15]
       }
   }
   ```
   - 包含 5 輛車，每輛車 5 個欄位：`[presence, x, y, vx, vy]`。
   - `presence`：車輛存在旗標（1.0 或 0.0）。
   - `x, y, vx, vy`：歸一化後數值落在 $[-1.0, 1.0]$。
4. **Action Configuration**：
   - `DiscreteMetaAction`（5 種離散決策：左換道、閒置、右換道、加速、減速）。
5. **單步動態模擬 (`step`) 與彩現 (`render`)**：
   - 呼叫 `env.step(1)` 成功回傳 5-tuple（observation, reward, terminated, truncated, info），並可彩現 `(600, 600, 3)` RGB array。

---

## 4. Task D — 相容性綜合比對 (Compatibility Check)

| 比較項目 | 老師規格 / NN 骨架要求 | 資料集 (`train.npz`) 實測 | 環境 (`roundabout-v0`) 實測 | 相容判定 |
| :--- | :--- | :--- | :--- | :---: |
| **Input Shape** | 25 維 (`obs.reshape(1, 25)`) | `(35000, 25)` | `(5, 5)` 展平為 `(25,)` | **100% 相容** |
| **Output Shape** | 5 維機率分佈 (Softmax) | 5 個類別 (`label.npy`) | `Discrete(5)` (0~4) | **100% 相容** |
| **Output Activation** | `Softmax` | 類別標籤 0~4 | 離散動作選擇 (`argmax`) | **100% 相容** |
| **Model 儲存格式** | `.h5` (Keras HDF5) | N/A | TensorFlow 2.13.0 Keras | **100% 相容** |
| **特徵欄位排列** | 5 輛車 $\times$ 5 特徵 | `[presence, x, y, vx, vy]` | `[presence, x, y, vx, vy]` | **100% 相容** |
| **特徵數值區間** | 浮點數特徵 | 嚴格落在 $[-1.0, 1.0]$ | 經 features_range 歸一至 $[-1, 1]$ | **100% 相容** |
| **Label 欄位名稱** | 老師引導：`train_dataset['...']` | Key 名稱為 `'label'` | 0~4 整數動作代碼 | **100% 相容** |

> [!TIP]
> **重要確認**：實測抽樣驗證資料集特徵矩陣，每一輛車的第一個欄位均為 `1.0` 或 `0.0`（即 `presence` 旗標），其餘四個欄位嚴格落於 $[-1.0, 1.0]$，證實**資料集特徵與 highway-env 的 Kinematics 輸出完全對齊，無須亦不可做破壞性的二次標準化**。

---

## 5. 發現的問題與風險點分析 (Issues & Findings)

1. **極度不平衡的動作標籤（Class Imbalance）**：
   - 動作 2（右轉道）僅有 305 筆（0.87%），動作 0（左轉道）僅有 806 筆（2.30%）。
   - **風險**：若直接計算 Accuracy，模型全猜 4 或 3 即可能獲得超過 70% 準確率，但在實際圓環中無法變換車道。
   - **解法**：在 Phase 2 使用 `sklearn.utils.class_weight.compute_class_weight` 計算加權權重，傳入 `model.fit(..., class_weight=class_weight_dict)`。
2. **資料型別匹配**：
   - `train.npz` 中的 `data` 與 `label` 原始型別為 `float64`。
   - Keras 訓練建議特徵轉為 `float32`，標籤先轉為 `int` 再做 `to_categorical(..., num_classes=5)`。
3. **檔案存取相容性**：
   - 老師的 `NN_train.py` 寫法為 `np.load(root_path + '/dataset/train.npz')`。
   - 目前已透過 `np.savez` 將原始 `data.npy` 與 `label.npy` 打包為標準 `train.npz` 與 `validation.npz`（keys: `['data', 'label']`），確保老師程式碼完全相容。

---

## 6. 建議的 Preprocessing 與 Phase 2 策略

1. **特徵前處理**：
   ```python
   # 轉為 float32，形狀已是 (N, 25)，數值已在 [-1.0, 1.0]，直接使用
   train_data = train_dataset['data'].astype(np.float32)
   ```
2. **標籤前處理**：
   ```python
   # 標籤 key 為 'label'，轉為整數後進行 One-Hot Encoding
   train_label_raw = train_dataset['label'].astype(int).flatten()
   train_label = to_categorical(train_label_raw, num_classes=5)
   ```
3. **類別權重計算 (Class Weights)**：
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   classes = np.array([0, 1, 2, 3, 4])
   weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_label_raw)
   class_weight_dict = dict(enumerate(weights))
   ```

---

## 7. 階段結論：Phase 2 是否可以開始？

### 結論：<font color="green">**可以立即開始 Phase 2！ (Ready for Phase 2)**</font>

- **資料集**：`train.npz` (35,000 筆) 與 `validation.npz` (5,000 筆) 完整無缺失，特徵排列與環境 100% 吻合。
- **環境**：Docker、TensorFlow 2.13.0、Gymnasium 0.29.1、highway-env 1.8.2 經實測完全正常。
- **規格匹配**：25 維輸入 $\rightarrow$ 5 維輸出 (Softmax) $\rightarrow$ 儲存為 `.h5`，所有管線規格明確清晰。
