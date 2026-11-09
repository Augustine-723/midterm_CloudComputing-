# Highway-env Roundabout Demo Day 現場執行清單（Checklist）

請於 Demo 當天依序勾選與執行各階段項目，確保評估流程萬無一失。

---

## 階段一：主機與容器就緒檢查（Demo 開始前 10 分鐘）

- [ ] **1.1 開啟終端機並切換至專案目錄**
  ```bash
  cd /home/augustine/cloud_computing/midterm_roundabout
  ```
- [ ] **1.2 確認 Docker Image 存在**
  ```bash
  docker images | grep roundabout-midterm
  ```
  *(確認輸出包含 `roundabout-midterm:latest`)*
- [ ] **1.3 啟動容器並驗證基礎環境**
  ```bash
  docker run --rm -v $(pwd):/app -w /app roundabout-midterm:latest python demo/scripts/check_environment.py
  ```
  *(確認最後印出 `[✓] DEMO ENVIRONMENT VERIFICATION: ALL CHECKS PASSED`)*
- [ ] **1.4 啟動常駐容器進入互動介面**
  ```bash
  docker run --rm -it -v $(pwd):/app -w /app roundabout-midterm:latest bash
  ```
  *(保持此終端機視窗開啟，後續所有操作均在此視窗執行)*

---

## 階段二：助教提供檔案與資料放入（Demo 現場）

- [ ] **2.1 接收助教檔案**
  - 正式測試資料檔名：_________________（例如 `test.npz`）
  - 正式測試程式檔名：_________________（例如 `test.py`）
- [ ] **2.2 將助教檔案放入 `demo/incoming/`**
  ```bash
  # 範例：從下載資料夾複製
  cp /path/to/ta_files/* demo/incoming/
  ls -lh demo/incoming/
  ```
- [ ] **2.3 執行唯讀測試資料檢測**
  ```bash
  python demo/scripts/inspect_test_data.py
  ```
  - [ ] 確認樣本數 $N$：________ 筆
  - [ ] 確認觀測特徵可展平為 `(N, 25)`
  - [ ] 確認無異常之 NaN 或 Inf

---

## 階段三：執行正式測試與記錄（正式評分）

- [ ] **3.1 詢問助教指定的執行指令**
  - 助教標準指令：_____________________________________________________
- [ ] **3.2 執行正式評估 Runner（自動記錄日誌與模型 SHA）**
  - 若為標準參數：
    ```bash
    python demo/scripts/run_official_test.py
    ```
  - 若助教要求指定指令：
    ```bash
    python demo/scripts/run_official_test.py --cmd "助教指定之指令"
    ```
- [ ] **3.3 確認測試程式正常結束**
  - [ ] Exit Code 為 0
  - [ ] 記錄正式 Test Accuracy：____________ %
  - [ ] 記錄正式 Test Loss：____________
- [ ] **3.4 確認模型完整性檢查通過**
  - [ ] `[✓] Model Integrity: PASS`（證明模型未被覆蓋或修改）
  - [ ] SHA-256 確認：`ccc4b358d95e9b56805ef8244eb7247d60f9a5d7bc5e5340c197652fef837e02`

---

## 階段四：進階加分項目展示（若助教詢問）

- [ ] **4.1 環境模擬測試展示 (`roundabout-v0` 100 Episodes)**
  ```bash
  python submission/NN_test.py
  ```
  *(輸出 10 個整數 Round 回報，如 `[99, 96, 94, ...]`)，證明可於真實 Highway-env 模擬器推論)*
- [ ] **4.2 客製化 GradientTape 訓練展示（進階加分 A，最高 10 分）**
  - 參考指南：[`reports/advanced_bonus_ta_demo_guide.md`](file:///home/augustine/cloud_computing/midterm_roundabout/reports/advanced_bonus_ta_demo_guide.md) Part A
  - 快速 Smoke Test（輸出至暫存區）：
    ```bash
    python submission/train_gradient_tape.py --epochs 1 --output /tmp/demo_gt.h5
    ```
  - 重點展示：禁絕 `model.fit()`、`with tf.GradientTape() as tape:`、`tape.gradient()`、`optimizer.apply_gradients()`
- [ ] **4.3 兩邏輯 CPU 資料並行展示（進階加分 B，最高 10 分）**
  - 參考指南：[`reports/advanced_bonus_ta_demo_guide.md`](file:///home/augustine/cloud_computing/midterm_roundabout/reports/advanced_bonus_ta_demo_guide.md) Part B
  - 快速 Smoke Test（輸出至暫存區）：
    ```bash
    python submission/train_distributed.py --epochs 1 --output /tmp/demo_dist.h5
    ```
  - 重點展示：`MirroredStrategy`、印出 `Replica IDs [0, 1]`、`sharded batch sizes [32, 32]`、跨設備權重同步斷言 `assert_replicas_synchronized`

---

## 階段五：結果封存與收尾

- [ ] **5.1 檢查 `demo/outputs/` 產出**
  ```bash
  ls -la demo/outputs/
  ```
- [ ] **5.2 打包本次 Demo 評估所有日誌與結果**
  ```bash
  tar -czvf demo_final_results_$(date +%Y%m%d_%H%M).tar.gz demo/outputs/
  ```
- [ ] **5.3 退出容器並安全清理暫存檔**
  ```bash
  exit
  ```

