# Highway-env Roundabout 期中專題：Demo 現場離線緊急故障排除手冊
**Offline Emergency Troubleshooting Guide**

本手冊專為 Demo 當天在無網路連線或無 AI 輔助環境下設計。手冊內所有指令均經過 Ubuntu 主機與 Docker 實境驗證。

---

## 1. 現場操作黃金準則與提示符號規範

### 1.1 提示符號規範（嚴格區分主機與容器）
- `[Host]$`：代表在 **Ubuntu 實體主機終端機** 下執行的指令。
- `[Container]#`：代表在 **Docker 容器內部（`/app` 目錄）** 下執行的指令。

### 1.2 故障排除三大禁止原則（嚴禁踩雷）
1. **嚴禁刪除既有映像檔與容器：** 切勿執行 `docker rmi` 或任意 `docker rm -f`，避免遺失已配置好之 Python 3.10 + TensorFlow 2.13.0 環境。
2. **嚴禁修改原始交件模型與 ZIP：** 切勿覆蓋 `submission/YOURMODEL.h5` 或重新打包 `midterm_roundabout_submission.zip`。
3. **嚴禁竄改助教評分邏輯：** 助教測試腳本是客觀計分依據，若有維度不合應透過包裝器或轉換器轉接，絕不可修改其 Accuracy 判斷式。

---

## 2. 基礎環境感知與狀態診斷指令

當 Demo 現場出現任何非預期狀況時，請立即依序執行以下 4 個基本診斷指令：

### 2.1 確認目前所在路徑
```bash
[Host]$ pwd
# 正確預期輸出：/home/augustine/cloud_computing/midterm_roundabout
```
*若不在專案根目錄，請立即執行：*
```bash
[Host]$ cd /home/augustine/cloud_computing/midterm_roundabout
```

### 2.2 確認專案核心檔案完整性
```bash
[Host]$ ls -lh submission/YOURMODEL.h5 midterm_roundabout_submission.zip
# 預期輸出：YOURMODEL.h5 約 172K (175,656 bytes)，ZIP 約 236K (241,098 bytes)
```

### 2.3 檢查 Docker 容器運作狀態
```bash
[Host]$ docker ps -a | grep roundabout
```
**狀態解讀：**
- 顯示 `Up X days` 或 `Up X minutes`：代表容器正在背景運行，可直接以 `docker exec` 進入。
- 顯示 `Exited (...)`：代表容器已停止，需重新啟動或以 `docker run` 啟動新容器。
- 輸出為空：代表尚未建立名為 `roundabout-midterm` 之容器，但有映像檔可用。

### 2.4 檢查 Docker 映像檔是否存在
```bash
[Host]$ docker images | grep roundabout-midterm
# 預期輸出：roundabout-midterm   latest   b5b5ef3aee31   1.76GB
```

---

## 3. 常見現場故障情境對照表與處置 SOP

### 情境 1：Docker Daemon 無法連線 / 權限不足
- **錯誤訊息特徵：**
  ```text
  Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
  # 或者：
  dial unix /var/run/docker.sock: connect: permission denied
  ```
- **可能成因：** Docker 守護行程未啟動，或當前終端機無存取 docker.sock 權限。
- **診斷指令：**
  ```bash
  [Host]$ sudo systemctl status docker
  ```
- **具體解決步驟：**
  1. 啟動 Docker 服務：
     ```bash
     [Host]$ sudo systemctl restart docker
     ```
  2. 若為權限問題，指令前補上 `sudo` 或暫時賦權：
     ```bash
     [Host]$ sudo docker ps
     # 或臨時允許當前使用者存取 socket：
     [Host]$ sudo chmod 666 /var/run/docker.sock
     ```

---

### 情境 2：容器停止或無法進入 (`roundabout-midterm`)
- **錯誤訊息特徵：**
  ```text
  Error response from daemon: Container ... is not running
  ```
- **可能成因：** 背景常駐的 `roundabout-midterm` 容器因重開機或記憶體不足而關閉。
- **具體解決步驟（二選一）：**
  - **方案 A（重啟既有容器）：**
    ```bash
    [Host]$ docker start roundabout-midterm
    [Host]$ docker exec -it roundabout-midterm bash
    ```
  - **方案 B（立即啟動乾淨臨時容器，最推薦）：**
    ```bash
    [Host]$ cd /home/augustine/cloud_computing/midterm_roundabout
    [Host]$ docker run --rm -it -v $(pwd):/app -w /app roundabout-midterm:latest bash
    ```

---

### 情境 3：容器名稱衝突 (`Conflict. The container name is already in use`)
- **錯誤訊息特徵：**
  ```text
  docker: Error response from daemon: Conflict. The container name "/roundabout-midterm" is already in use by container "...".
  ```
- **可能成因：** 嘗試使用 `--name roundabout-midterm` 啟動新容器，但舊容器尚未移除。
- **具體解決步驟：**
  直接使用不指定固定名稱的 `--rm` 參數啟動（隨機指派名稱，結束自動清理）：
  ```bash
  [Host]$ docker run --rm -it -v $(pwd):/app -w /app roundabout-midterm:latest bash
  ```

---

### 情境 4：檔案或目錄找不到 (`FileNotFoundError`)
- **錯誤訊息特徵：**
  ```text
  FileNotFoundError: Could not locate train.npz in candidate directories
  # 或：
  No such file or directory: 'submission/YOURMODEL.h5'
  ```
- **可能成因：** 在容器內執行時未處於 `/app` 目錄，或掛載路徑錯誤。
- **診斷指令：**
  ```bash
  [Container]# pwd
  [Container]# ls -la
  ```
- **具體解決步驟：**
  1. 切換至 `/app` 根目錄：
     ```bash
     [Container]# cd /app
     ```
  2. 檢查掛載目錄下是否存在 `submission/` 與 `dataset/`：
     ```bash
     [Container]# ls -la submission/YOURMODEL.h5
     [Container]# ls -la dataset/train.npz dataset/validation.npz
     ```
  3. 若在 host 端掛載錯誤，退出容器重新掛載：
     ```bash
     [Host]$ cd /home/augustine/cloud_computing/midterm_roundabout
     [Host]$ docker run --rm -it -v $(pwd):/app -w /app roundabout-midterm:latest bash
     ```

---

### 情境 5：缺少 Python 套件 (`ModuleNotFoundError: No module named '...'`)
- **錯誤訊息特徵：**
  ```text
  ModuleNotFoundError: No module named 'tqdm'
  # 或：No module named 'pandas', 'seaborn' 等
  ```
- **可能成因：** 助教提供的評估程式額外引用了非基礎依賴套件。
- **具體解決步驟：**
  1. 在運行中的容器內直接安裝（不破壞原始映像檔）：
     ```bash
     [Container]# pip install <套件名稱>
     # 範例：
     [Container]# pip install tqdm pandas matplotlib
     ```
  2. 若現場完全無外網連線（Offline）：
     - 檢查助教腳本，若 `tqdm` 僅用於進度條，告知助教暫時移除 `from tqdm import tqdm` 並將 `tqdm(range(...))` 改為原生 `range(...)`。
     - 若為評估必要運算庫，請助教提供 `.whl` 檔或允許主機暫時連線熱點安裝。

---

### 情境 6：模型載入失敗 (`Unable to open file` / `Custom objects`)
- **錯誤訊息特徵：**
  ```text
  ValueError: Unknown layer: '...' 或 OSError: Unable to open file (file signature not found)
  ```
- **可能成因：** 助教評分程式使用了高階 `compile=True` 嘗試載入非標準優化器，或檔案毀損。
- **診斷指令：**
  ```bash
  [Container]# sha256sum submission/YOURMODEL.h5
  # 正確雜湊值必須是：ccc4b358d95e9b56805ef8244eb7247d60f9a5d7bc5e5340c197652fef837e02
  ```
- **具體解決步驟：**
  1. 向助教說明本專案模型是以 TensorFlow 2.13.0 標準 HDF5 格式儲存。
  2. 請助教在載入模型時加入 `compile=False`：
     ```python
     import tensorflow as tf
     model = tf.keras.models.load_model("submission/YOURMODEL.h5", compile=False)
     ```
  3. 若 `submission/YOURMODEL.h5` 遭誤刪或被改動，立即從備用 ZIP 解壓還原：
     ```bash
     [Host]$ unzip -o midterm_roundabout_submission.zip submission/YOURMODEL.h5
     ```

---

### 情境 7：測試資料維度不符 (`Incompatible input shape`)
- **錯誤訊息特徵：**
  ```text
  ValueError: Input 0 of layer "dense" is incompatible with the layer: expected min_ndim=2, found ndim=3. Full shape received: (None, 5, 5)
  ```
- **可能成因：** 助教的 `test.npz` 未展平觀測特徵（5 輛車 $\times$ 5 屬性保留為 3D 矩陣 `(N, 5, 5)`），而模型預期輸入為 2D 向量 `(None, 25)`。
- **診斷指令：**
  ```bash
  [Container]# python demo/scripts/inspect_test_data.py demo/incoming/<助教資料檔名>.npz
  ```
- **具體解決步驟：**
  1. 檢視診斷輸出，確認資料鍵名與形狀（如 `data: (N, 5, 5)`）。
  2. 依據題目規範，向助教說明模型輸入為 25 維連續特徵，請助教在其測試程式加入展平前處理：
     ```python
     # 助教測試程式內轉換：
     obs = obs.reshape(-1, 25).astype(np.float32)
     ```
  3. 若助教要求我們自行適配，可使用標準轉接寫法：
     ```python
     x_test = test_data["data"].reshape(-1, 25).astype(np.float32)
     ```

---

### 情境 8：遠端連線（RDP / SSH）中斷或終端機卡死
- **錯誤訊息特徵：** 畫面凍結、輸入無反應、連線中斷重連後找不到正在跑的測試。
- **可能成因：** 網路暫態中斷或長時間模擬測試導致終端機超時。
- **具體解決步驟：**
  1. 重新登入後，先確認 Docker 容器是否仍在背景執行：
     ```bash
     [Host]$ docker ps
     ```
  2. 若容器仍在執行，重新 attach 進去：
     ```bash
     [Host]$ docker exec -it roundabout-midterm bash
     ```
  3. 若擔心測試中途斷線，建議使用 `nohup` 或查看 `demo/outputs/` 的日誌：
     ```bash
     [Container]# tail -f demo/outputs/run_*/stdout.log
     ```

---

### 情境 9：行程被強制終止 (`Killed` / Out of Memory)
- **錯誤訊息特徵：**
  ```text
  Killed
  # 或 Exit Code 137
  ```
- **可能成因：** 測試資料筆數過大，或前向傳播一次將所有測試資料轉為 NumPy 陣列塞爆記憶體。
- **具體解決步驟：**
  若資料筆數極大（如超過 10 萬筆），請助教以批次（Batch）分段推論：
  ```python
  batch_size = 512
  all_preds = []
  for i in range(0, len(x_test), batch_size):
      batch_x = x_test[i:i + batch_size]
      probs = model(batch_x, training=False).numpy()
      all_preds.append(np.argmax(probs, axis=1))
  final_preds = np.concatenate(all_preds, axis=0)
  ```

---

## 4. 進階加分項目現場展示程序（已驗證）

若助教要求現場展示客製化訓練迴圈或分散式資料並行，請依以下指令安全展示（均已實測，保證不覆蓋原始模型）：

### 4.1 客製化 GradientTape 訓練展示（加分項目 A，最高 10 分）
- **執行指令（安全輸出至 `/tmp/demo_gt.h5`）：**
  ```bash
  [Container]# python submission/train_gradient_tape.py --epochs 1 --output /tmp/demo_gt.h5
  ```
- **預期終端輸出證據：**
  ```text
  ======================================================================
   ADVANCED BONUS: CUSTOM TRAINING LOOP WITH TF.GRADIENTTAPE
  ======================================================================
  Loaded Train: 35000 samples | Val: 5000 samples
  Epoch 01/01 (1.53s) — loss: 0.6632 - acc: 73.22% | val_loss: 0.4989 - val_acc: 79.24% [Saved -> val_loss: 0.49889]

  Model saved to: /tmp/demo_gt.h5
  ```
- **口試核心重點：**
  - 指出代碼第 193 行：`with tf.GradientTape() as tape:` 記錄前向傳播。
  - 指出代碼第 196 行：`grads = tape.gradient(loss, model.trainable_variables)` 逆向自動微分。
  - 指出代碼第 197 行：`optimizer.apply_gradients(...)` 顯式更新權重。
  - 強調全檔 **0 次** 呼叫 `model.fit()` 或 `model.evaluate()`。

---

### 4.2 兩邏輯 CPU 資料並行展示（加分項目 B，最高 10 分）
- **執行指令（安全輸出至 `/tmp/demo_dist.h5`）：**
  ```bash
  [Container]# python submission/train_distributed.py --epochs 1 --output /tmp/demo_dist.h5
  ```
- **預期終端輸出證據（關鍵實測證據）：**
  ```text
  ======================================================================
   ADVANCED BONUS: SYNCHRONOUS DATA PARALLELISM (MIRROREDSTRATEGY)
  ======================================================================
  Logical CPU Devices:   ['/device:CPU:0', '/device:CPU:1']
  Replicas in Sync:      2
  Global Batch Size:     64 (32 per replica)
  [EVIDENCE] Replica IDs: [0, 1]; sharded batch sizes: [32, 32]; global batch: 64
  [EVIDENCE] All 6 trainable variables are synchronized across 2 replicas.
  Epoch 01/01 (3.99s) — loss: 0.6478 - acc: 73.15% | val_loss: 0.4866 - val_acc: 79.18% [Saved -> val_loss: 0.48660]

  Model saved to: /tmp/demo_dist.h5
  ```
- **口試核心重點：**
  - 指出代碼第 43 行：`tf.config.set_logical_device_configuration` 虛擬化 2 個 Logical CPU。
  - 指出代碼第 118 行：`tf.distribute.MirroredStrategy` 同步鏡像策略。
  - 指出代碼第 238 行：`strategy.run(step_fn)` 並行調度。
  - 指出代碼第 248 行：`assert_replicas_synchronized()` 逐點比對兩設備權重，數學級保證一致。
  - 誠實說明：在單一 CPU 上因鎖定與通訊開銷（0.39s vs 2.05s），本實作重在驗證同步資料並行架構之正確性，無虛構加速。

---

## 5. 正式測試評估執行 SOP（待助教測試程式確認）

> [!WARNING]
> **重要聲明：** 助教之正式測試資料（如 `test.npz`）與測試程式（如 `test.py`）尚未釋出。以下步驟為自適應評估 SOP，實際指令須依據助教公佈之參數規格為準。

### Step 1：放入助教檔案
將助教檔案放置於 `demo/incoming/`：
```bash
[Host]$ cp /path/to/ta_files/* demo/incoming/
```

### Step 2：執行唯讀資料格式診斷
```bash
[Container]# python demo/scripts/inspect_test_data.py
```
- 確認是否檢測到 25 維特徵與標籤陣列。
- 確認無異常 NaN / Inf。

### Step 3：執行自適應評估 Runner
- **情況 A（助教提供標準參數腳本）：**
  ```bash
  [Container]# python demo/scripts/run_official_test.py
  ```
- **情況 B（助教要求特定 CLI 旗標）：**
  ```bash
  [Container]# python demo/scripts/run_official_test.py --script demo/incoming/test.py --data demo/incoming/test.npz --model submission/YOURMODEL.h5
  ```
- **情況 C（助教要求完全自訂之終端指令）：**
  ```bash
  [Container]# python demo/scripts/run_official_test.py --cmd "python demo/incoming/test.py --eval_model submission/YOURMODEL.h5"
  ```

### Step 4：評估完成確認
1. 檢視終端輸出之 Test Accuracy。
2. 確認終端輸出 `[✓] Model Integrity: PASS`（證明模型未遭改動）。
3. 日誌自動完整儲存於 `demo/outputs/run_YYYYMMDD_HHMMSS/`。

---

## 6. 模型與檔案 SHA-256 基準雜湊表

若現場發生檔案混淆或疑慮，請以此表執行雜湊校驗比對：

```bash
[Host]$ sha256sum submission/YOURMODEL.h5 submission/models/* midterm_roundabout_submission.zip
```

| 檔案名稱 | 正確 SHA-256 基準雜湊值 | 角色說明 |
| :--- | :---: | :--- |
| `submission/YOURMODEL.h5` | `ccc4b358d95e9b56805ef8244eb7247d60f9a5d7bc5e5340c197652fef837e02` | 正式交件模型 (Baseline Standard) |
| `submission/models/distributed_cpu_best.h5` | `56ac256b6211f8b8d78accb2bc53e5734be7820d4ff227fef7676104c3539101` | 分散式並行佐證模型 (Val Acc 94.60%) |
| `submission/models/gradient_tape_best.h5` | `fe83d4fdd89dfae83c406ced5a4fadd94725199937adab3fddfb5ec017428afd` | GradientTape 佐證模型 (Val Acc 94.12%) |
| `midterm_roundabout_submission.zip` | `29abbc0bbbcdf2112b52cc2f906bef583039b7f2caaa674f40516ffb4ab72205` | 正式交件壓縮包 |

