# Highway-Env Roundabout 期中專題 Demo Day 評估指南

本手冊提供在 Demo 當天拿到助教提供之正式測試資料（如 `test.npz`）與測試程式（如 `test.py`）時，於 Ubuntu Docker 環境下進行評估與展示的完整操作步驟。

---

## 1. Demo 前要開啟哪些程式

在 Demo 開始前，請確認主機就緒：
1. **開啟 Ubuntu Terminal**（或 VS Code 終端機）。
2. **切換至專案根目錄：**
   ```bash
   cd /home/augustine/cloud_computing/midterm_roundabout
   ```
3. **確認 Docker 服務正常運行：**
   ```bash
   docker info > /dev/null 2>&1 && echo "[✓] Docker is running" || echo "[!] Docker is not running!"
   ```

---

## 2. 如何確認 Docker Image 存在

執行以下指令檢查期中專用映像檔 `roundabout-midterm:latest`：
```bash
docker images | grep roundabout-midterm
```
**預期輸出：**
```text
roundabout-midterm   latest   b5b5ef3aee31   1.76GB
```
若映像檔已存在，可直接使用，**切勿重新 build 或更動 Dockerfile**。

---

## 3. 如何啟動容器

我們採用**安全掛載**模式啟動容器：
- 將本專案目錄掛載至容器內的 `/app`。
- 設定容器工作目錄為 `/app`。

### 方式 A：互動式進入容器終端機（推薦現場 Demo 使用）
```bash
docker run --rm -it \
  -v /home/augustine/cloud_computing/midterm_roundabout:/app \
  -w /app \
  roundabout-midterm:latest bash
```

### 方式 B：單行指令執行（不進入互動介面）
```bash
docker run --rm \
  -v /home/augustine/cloud_computing/midterm_roundabout:/app \
  -w /app \
  roundabout-midterm:latest python demo/scripts/check_environment.py
```

---

## 4. 助教提供檔案時應放在哪裡

當助教提供正式測試資料與評分程式時，請放置於專屬的 `incoming/` 目錄：

```text
/home/augustine/cloud_computing/midterm_roundabout/demo/incoming/
├── test.npz     <-- 助教提供的正式測試資料
└── test.py      <-- 助教提供的正式評分程式 (若有提供)
```

**快速複製指令範例：**
```bash
# 假設助教檔案在隨身碟或 ~/Downloads：
cp /media/$USER/*/test.npz demo/incoming/
cp /media/$USER/*/test.py demo/incoming/
```

> [!IMPORTANT]
> - `demo/incoming/` 在取得助教檔案前保持空白，嚴禁自行偽造測試集。
> - 助教檔案放入後，原始交件模型 `submission/YOURMODEL.h5` 與交件包絕不受影響。

---

## 5. 如何檢查測試資料

助教資料放入 `demo/incoming/` 後，先執行**唯讀診斷腳本**，檢查形狀與數值健康度：

```bash
# 在 Docker 內執行：
python demo/scripts/inspect_test_data.py
```
*(若檔案在其他路徑，可直接帶入路徑：`python demo/scripts/inspect_test_data.py /path/to/test.npz`)*

**腳本將自動輸出：**
- 內部陣列鍵名（如 `'data'`, `'label'`）
- 資料樣本數 $N$、矩陣形狀（如 `(N, 25)`）
- 資料型態（`float32`, `int32`）
- 是否含有 NaN 或 Inf
- 是否與 `YOURMODEL.h5` 之輸入形狀 `(None, 25)` 完全相容
- 標籤類別是否介於 0～4 之間（且絕不印出全體測試標籤，保障評分私密性）

---

## 6. 如何執行正式測試程式

依據助教提供的測試介面，我們提供三種自適應執行方式：

### 情況一：助教提供標準腳本（放置於 incoming/）
自動偵測 `demo/incoming/` 內的 `.py` 與 `.npz`：
```bash
python demo/scripts/run_official_test.py
```

### 情況二：助教指定自訂執行參數
若助教測試腳本要求特定的 CLI 旗標（如 `--model` 與 `--data`）：
```bash
python demo/scripts/run_official_test.py \
  --script demo/incoming/test.py \
  --data demo/incoming/test.npz \
  --model submission/YOURMODEL.h5
```

### 情況三：助教要求自訂命令格式
透過 `--cmd` 參數傳遞任意完整執行命令：
```bash
python demo/scripts/run_official_test.py \
  --cmd "python demo/incoming/test.py --test_data demo/incoming/test.npz --model_path submission/YOURMODEL.h5"
```

> [!NOTE]
> `run_official_test.py` 在執行前與執行後會自動比對模型的 SHA-256，證明模型在評估過程中未被竄改或重新訓練。

---

## 7. 如何查看 Accuracy

1. **若助教腳本直接輸出：**
   終端機將即時顯示測試程式輸出的 Test Accuracy 與 Loss。
2. **日誌查閱：**
   所有終端輸出會同步寫入：
   `demo/outputs/run_YYYYMMDD_HHMMSS/stdout.log`
3. **離線驗證集對比（供對照參考，非正式測試成績）：**
   若需向助教展示模型在原驗證集之表現（94.64%），可執行：
   ```bash
   python demo/scripts/verify_pre_demo.py
   ```

---

## 8. 如何保存測試結果

每次執行 `run_official_test.py`，系統會自動在 `demo/outputs/` 建立時間戳記目錄：
```text
demo/outputs/run_YYYYMMDD_HHMMSS/
├── stdout.log         # 助教測試程式之完整標準輸出
├── stderr.log         # 錯誤日誌 (若有)
└── run_summary.json   # 包含執行耗時、Exit Code、模型 SHA 雜湊校驗結果
```
**將結果打包保存指令：**
```bash
tar -czvf demo_results_$(date +%Y%m%d_%H%M%S).tar.gz demo/outputs/
```

---

## 9. 遇到 Missing Module 怎麼處理

若助教提供的評估程式使用了容器內未預先安裝的 Python 套件（如 `tqdm`, `pandas`, `seaborn` 等）：

1. **直接在運行中的容器臨時安裝（不修改映像檔）：**
   ```bash
   pip install <package_name>
   ```
2. **若無對外網路連線（離線環境）：**
   檢視助教程式中該模組的使用位置。若是 `tqdm`，通常僅為進度條顯示，可請求助教以簡單 `for` 迴圈替代；若是評估計算核心，請向助教反映補裝該依賴。

---

## 10. 遇到 Model Shape Mismatch 怎麼處理

若助教測試程式載入模型時出現形狀不合錯誤：
1. **輸入維度形狀不合：**
   - 我們的模型接受形狀為 `(None, 25)`。
   - 若助教資料為 `(N, 5, 5)`（未展平的 5 輛車 × 5 屬性），請向助教確認是否需在測試腳本加入展平：
     ```python
     obs = obs.reshape(-1, 25).astype(np.float32)
     ```
2. **輸出維度形狀不合：**
   - 我們的模型輸出為 5 個動作的機率分佈 `(None, 5)`。
   - 預測動作需取最大值索引：`action = np.argmax(probs, axis=1)`。
   - 若助教程式預期單一整數向量，確認其是否有呼叫 `argmax`。
3. **Keras 版本載入問題：**
   - 我們的模型由 TensorFlow 2.13.0 訓練儲存為 HDF5 (`.h5`) 格式。
   - 建議助教載入時使用：
     ```python
     import tensorflow as tf
     model = tf.keras.models.load_model("submission/YOURMODEL.h5", compile=False)
     ```
   - 加上 `compile=False` 可徹底排除任何損失函數或優化器自訂物件相容性問題。

