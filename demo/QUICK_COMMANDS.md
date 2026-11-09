# Roundabout Demo Day — 一頁式快速指令表（Quick Commands）

> **提示：** `[Host]$` 代表在 Ubuntu 主機終端執行，`[Container]#` 代表在 Docker 容器內執行。

---

## 1. 30 秒快速啟動與健康檢查（Demo 開場必備）

```bash
# 1. 切換至專案根目錄
[Host]$ cd /home/augustine/cloud_computing/midterm_roundabout

# 2. 一鍵執行環境與模型健康檢查（全 PASS 即可安心 Demo）
[Host]$ docker run --rm -v $(pwd):/app -w /app roundabout-midterm:latest python demo/scripts/check_environment.py

# 3. 進入 Docker 容器（保持此視窗常駐，後續指令都在容器內執行）
[Host]$ docker run --rm -it -v $(pwd):/app -w /app roundabout-midterm:latest bash
```

---

## 2. 助教檔案放入後的評估三步驟（正式評分）

```bash
# Step 1：[Host 另開視窗] 將助教提供的 test.npz 與 test.py 放入 incoming 資料夾
[Host]$ cp /path/to/ta_files/* demo/incoming/

# Step 2：[Container 視窗] 唯讀診斷測試資料格式（確認 sample 數、無 NaN/Inf、特徵相容）
[Container]# python demo/scripts/inspect_test_data.py

# Step 3：[Container 視窗] 執行正式測試評估（自動記錄日誌並核對模型未遭竄改）
[Container]# python demo/scripts/run_official_test.py
# 若助教指定自訂指令：
[Container]# python demo/scripts/run_official_test.py --cmd "助教給的指令"
```

---

## 3. 進階加分項目現場安全展示（絕不覆蓋模型）

```bash
# A. 模擬器推論展示（100 回合無碰撞，輸出 10 個整數 Round 回報）
[Container]# python submission/NN_test.py

# B. 客製化 GradientTape 訓練展示（1-Epoch Smoke Test，輸出至 /tmp）
[Container]# python submission/train_gradient_tape.py --epochs 1 --output /tmp/demo_gt.h5

# C. 兩邏輯 CPU 資料並行展示（印出 Replica IDs [0, 1]、Shard [32, 32] 與權重同步斷言）
[Container]# python submission/train_distributed.py --epochs 1 --output /tmp/demo_dist.h5
```

---

## 4. 現場常見錯誤一秒急救指令（Quick Fix）

| 遇到錯誤情況 | 終端急救單行指令 |
| :--- | :--- |
| **容器名稱衝突 / 容器進不去** | `docker run --rm -it -v $(pwd):/app -w /app roundabout-midterm:latest bash` |
| **Docker Daemon 權限被擋** | `sudo chmod 666 /var/run/docker.sock` |
| **缺少未預期 Python 套件** | `pip install <package_name>`（容器內直接安裝） |
| **模型被助教腳本報錯無法載入** | 請助教在 `load_model` 時加上參數：`compile=False` |
| **測試特徵維度報錯 (None, 5, 5)** | 請助教在讀入資料後加上展平：`obs = obs.reshape(-1, 25).astype(np.float32)` |
| **終端機連線中斷重連後找日誌** | `cat demo/outputs/run_*/stdout.log` |

---

## 5. 核心模型與交件 ZIP 雜湊查核（SHA-256）

```bash
[Host]$ sha256sum submission/YOURMODEL.h5 submission/models/* midterm_roundabout_submission.zip
```
- **正式模型 `YOURMODEL.h5`：** `ccc4b358d95e9b56805ef8244eb7247d60f9a5d7bc5e5340c197652fef837e02`
- **分散式模型 `distributed_cpu_best.h5`：** `56ac256b6211f8b8d78accb2bc53e5734be7820d4ff227fef7676104c3539101`
- **GradientTape 模型 `gradient_tape_best.h5`：** `fe83d4fdd89dfae83c406ced5a4fadd94725199937adab3fddfb5ec017428afd`
- **正式交件 ZIP `midterm_roundabout_submission.zip`：** `29abbc0bbbcdf2112b52cc2f906bef583039b7f2caaa674f40516ffb4ab72205`

