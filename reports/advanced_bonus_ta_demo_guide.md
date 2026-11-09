# Highway-env Roundabout 期中專題：進階加分項目口試與 TA Demo 指南

**專案名稱：** Highway-env Roundabout 自律駕駛動作決策神經網路  
**加分項目：** A. 客製化訓練迴圈（最高 10 分）+ B. 分散式資料並行訓練（最高 10 分）  
**目標對象：** 學生口試準備、助教（TA）現場展示、技術答辯專用  
**對應檔案：**
- [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py)
- [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py)
- [`submission/NN_test.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/NN_test.py)

---

## Part A：客製化訓練迴圈（tf.GradientTape, 最高 10 分）

### Q1. 什麼是 tf.GradientTape？它的核心運作原理是什麼？
- **30 秒口頭回答：**  
  「`tf.GradientTape` 是 TensorFlow 提供的低階自動微分上下文管理器。在 `with tf.GradientTape() as tape:` 區塊內，TensorFlow 會在計算圖中動態記錄所有涉及可訓練變數（Trainable Variables）的前向傳播運算步驟；當運算完成後，呼叫 `tape.gradient(loss, variables)`，它便會利用微積分連鎖律執行逆向自動微分（Reverse-mode Autodiff / Vector-Jacobian Product），精確計算出純量損失對每個權重與偏差張量的一階偏導數梯度向量。」
- **對應檔案與函式：** [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py) 之 `train_step(x_b, y_b)`
- **程式碼行號：** 第 193～196 行

---

### Q2. 為什麼本專題完全不使用 Keras 高階 API（如 model.fit()）？@tf.function 的角色是什麼？
- **30 秒口頭回答：**  
  「因為老師在進階加分規定中明確要求：『不使用 Keras 高階 API，改用 `tf.GradientTape` 自行實作客製化訓練、驗證與測試』。使用 `model.fit()` 會將前向傳播、反向傳播與權重更新完全封裝。我們改寫為底層的 `train_step` 與 `val_step`，能完全掌控前向運算與梯度計算。同時，我們在函式上方標記 `@tf.function`，利用 Python Tracing 與 AutoGraph 將 Python 代碼構建為 TensorFlow 靜態計算圖（FunctionGraph），大幅消除 Python 解譯器的逐行直譯開銷（註：此為 Graph Tracing 機制，非 XLA JIT 編譯）。」
- **對應檔案與函式：** [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py) 之 `train_step`
- **程式碼行號：** 第 6～16 行（Docstring 說明）及第 178～185 行（@tf.function 宣告）

---

### Q3. 前向傳播（Forward Pass）在哪裡執行？為什麼帶入 training=True？
- **30 秒口頭回答：**  
  「前向傳播是在 `train_step` 函式內的第 194 行，直接呼叫 `preds = model(x_b, training=True)`。帶入 `training=True` 是告訴神經網路目前處於訓練模式，這在架構包含 Dropout 或 Batch Normalization 時特別關鍵，能確保隨機失活與批次統計量正確更新，使前向推論符合訓練期的隨機正則化行為。」
- **對應檔案與函式：** [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py) 之 `train_step(x_b, y_b)`
- **程式碼行號：** 第 194 行

---

### Q4. 損失函數（Loss）是如何定義與計算的？
- **30 秒口頭回答：**  
  「我們使用多分類交叉熵損失 `tf.keras.losses.SparseCategoricalCrossentropy()`。因為我們的標籤是純整數（0 到 4 分別代表 5 種駕駛動作），使用 Sparse 版本可以直接輸入整數標籤，省去轉為 One-Hot 編碼的記憶體開銷。在前向傳播後，透過 `loss = loss_fn(y_b, preds)` 計算預測機率分佈與真實動作索引之間的交叉熵數值，數值越小代表預測越準確。」
- **對應檔案與函式：** [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py) 之 `main()` 與 `train_step(x_b, y_b)`
- **程式碼行號：** 第 167 行（實例化）與第 195 行（計算損失）

---

### Q5. 梯度（Gradients）是如何計算的？
- **30 秒口頭回答：**  
  「梯度是透過 `grads = tape.gradient(loss, model.trainable_variables)` 顯式計算出來的。Tape 會調用計算圖中儲存的中間張量，根據微積分連鎖律計算純量 `loss` 對模型中全部 11,909 個可訓練參數（Dense 層的 weights 與 biases）的一階偏導數 $\\nabla_W L$。這完全跳過高階封裝，是純手工神經網路訓練的核心步驟。」
- **對應檔案與函式：** [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py) 之 `train_step(x_b, y_b)`
- **程式碼行號：** 第 196 行

---

### Q6. 模型權重是在哪裡更新的？使用什麼優化器？
- **30 秒口頭回答：**  
  「權重是在第 197 行透過 `optimizer.apply_gradients(zip(grads, model.trainable_variables))` 更新的。我們使用的是 Adam 優化器（Learning Rate = 0.001），它結合了動量（Momentum）與 RMSprop 自適應學習率。`zip` 函式將剛剛計算出的各層梯度與對應的權重變數配對傳入，優化器便會根據當前梯度更新模型權重。」
- **對應檔案與函式：** [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py) 之 `train_step(x_b, y_b)`
- **程式碼行號：** 第 165 行（Adam 初始化）與第 197 行（權重更新）

---

### Q7. 驗證步驟（val_step）與訓練步驟有何不同？為什麼不更新權重？
- **30 秒口頭回答：**  
  「有三大本質差異：第一，`val_step` 呼叫 `model(x_b, training=False)`，關閉訓練期特定行為；第二，它完全不使用 `tf.GradientTape()`，不記錄反向傳播圖，節省記憶體與運算；第三，驗證步驟絕對不呼叫 `apply_gradients`。因為驗證集的目的是以未參與訓練的獨立資料評估泛化能力，若在驗證集更新權重會造成嚴重的資料洩漏（Data Leakage）與過擬合。」
- **對應檔案與函式：** [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py) 之 `val_step(x_b, y_b)`
- **程式碼行號：** 第 203～216 行

---

### Q8. 測試環節（Testing Loop）是如何執行的？
- **30 秒口頭回答：**  
  「本專題的測試嚴格區分兩層架構：第一，在實際環境測試 [`submission/NN_test.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/NN_test.py) 中，模型直接以張量呼叫 `model(obs, training=False)` 進行 100 Episodes 的決策推論，完全禁絕 `model.predict()`；第二，針對離線資料集，因為課程僅釋出 train 與 validation 兩份公開資料，正式交件腳本專注於訓練與驗證，但我們在 [`scripts/evaluate_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/evaluate_gradient_tape.py) 中獨立實作了不依賴 Keras 高階 API 的測試腳本，完整計算混淆矩陣、Precision、Recall 與 Macro-F1。」
- **對應檔案與函式：** [`submission/NN_test.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/NN_test.py) 之 `main()` 與 [`scripts/evaluate_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/evaluate_gradient_tape.py) 之 `compute_metrics_direct()`
- **程式碼行號：** `NN_test.py` 第 51～53 行；`evaluate_gradient_tape.py` 第 32～105 行

---

### Q9. 早停機制（EarlyStopping）與 Checkpoint 是如何實作的？
- **30 秒口頭回答：**  
  「我們沒有使用 Keras 的 `EarlyStopping` Callback，而是以純 Python 手工實作狀態追蹤。我們監控每個 Epoch 的 `val_loss`：若當前損失比歷史最佳值改善超過 1e-4，則更新 `best_val_loss`、重置 `patience_cnt = 0` 並呼叫 `model.save()` 保存最佳權重；若連續 10 個 Epoch 沒有改善，則 `patience_cnt >= 10` 觸發 `break` 提前停止訓練，有效防止過擬合。」
- **對應檔案與函式：** [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py) 之 `main()`
- **程式碼行號：** 第 247～263 行

---

### Q10. tf.data.Dataset 的資料管線如何建立？Prefetch 的好處是什麼？
- **30 秒口頭回答：**  
  「我們使用 `tf.data.Dataset.from_tensor_slices` 載入記憶體陣列，接著透過 `shuffle(buffer_size=10000)` 打亂車輛時序連續性，再以 `batch(64)` 進行批次劃分。最後關鍵步驟是 `prefetch(tf.data.AUTOTUNE)`：它能讓 CPU 在 GPU/模型計算當前 Batch 時，在背景非同步預先載入下一個 Batch 的資料至記憶體，實現 I/O 與計算重疊（Overlapping），徹底消除資料讀取的等待延遲。」
- **對應檔案與函式：** [`submission/train_gradient_tape.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_gradient_tape.py) 之 `main()`
- **程式碼行號：** 第 137～156 行

---

## Part B：分散式資料並行訓練（Data Parallelism, 最高 10 分）

### Q1. 什麼是 Data Parallelism（資料並行）？它的架構與原理是什麼？
- **30 秒口頭回答：**  
  「Data Parallelism 是深度學習中最廣泛應用的分散式架構。它的原理是將完全相同的模型權重複製到多個計算設備（Replica）上，每個設備分配到全局批次（Global Batch）中的不同子集（Data Shard）。各設備各自獨立計算前向傳播與局部梯度，接著透過跨設備通訊（All-Reduce）將所有設備的梯度加總平均，最後每個設備同步更新相同的權重，達到多設備協同訓練的效果。」
- **對應檔案與函式：** [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py) 架構總覽
- **程式碼行號：** 第 6～16 行（Docstring）與第 210～239 行（分散式步函式）

---

### Q2. 什麼是 MirroredStrategy？為什麼使用它？
- **30 秒口頭回答：**  
  「`tf.distribute.MirroredStrategy` 是 TensorFlow 官方支援單機多設備（單機多 GPU 或單機多邏輯 CPU）的同步資料並行策略。它會在指定的計算設備上鏡像複製變數（MirroredVariable），並在背後管理資料分片與跨設備梯度聚合。採用同步模式的好處是梯度更新等價於單一設備以大批次訓練，數學邏輯嚴密，不會產生異步訓練中的過期梯度（Stale Gradients）問題。」
- **對應檔案與函式：** [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py) 之 `main()`
- **程式碼行號：** 第 118～125 行

---

### Q3. 為什麼只有一顆實體 CPU 卻能有兩個 Replica？
- **30 秒口頭回答：**  
  「這是利用 TensorFlow 的裝置虛擬化技術。在我們的實體主機上（Intel i5-9400 6 核心，無獨立 GPU），我們在 TensorFlow 執行期初始化前，調用 `tf.config.set_logical_device_configuration` 將這顆實體 CPU 劃分為兩個獨立的 Logical CPU Devices（`/device:CPU:0` 與 `/device:CPU:1`）。這讓 TensorFlow 將它們視為兩個獨立的硬體算力單元，從而能夠完整部署並驗證 `MirroredStrategy` 的多設備同步邏輯。」
- **對應檔案與函式：** [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py) 頂層配置
- **程式碼行號：** 第 41～45 行

---

### Q4. 全域批次大小（Global Batch Size = 64）如何分配給各個 Replica？
- **30 秒口頭回答：**  
  「我們建立 `batch(args.global_batch_size)` 的資料集後，傳入 `strategy.experimental_distribute_dataset(train_ds)`。TensorFlow 的分散式管線會自動將 Global Batch 64 等分為兩份，分配 32 筆給 Replica 0，32 筆給 Replica 1。如果遇到資料集末尾無法整除的批次（例如 35,000 除以 64 餘 56），它會自動均分 28 筆與 28 筆，確保每一筆樣本都精確參與訓練且各設備負載平衡。」
- **對應檔案與函式：** [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py) 之 `main()`
- **程式碼行號：** 第 152～164 行

---

### Q5. strategy.run() 在分散式訓練中扮演什麼角色？
- **30 秒口頭回答：**  
  「`strategy.run(step_fn, args=(dist_inputs,))` 是執行分散式運算的核心調度器。它會將分發後的輸入張量解包為各設備的 Shard，並在底層多個 Replica 執行緒上同時並行呼叫 `step_fn`。每個 Replica 在自己的上下文內執行前向傳播與反向微分，最後 `strategy.run` 會收集所有 Replica 的局部回傳值（如各設備的 loss 與 shard size），以 `PerReplica` 物件統一回傳。」
- **對應檔案與函式：** [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py) 之 `train_step(dist_inputs)`
- **程式碼行號：** 第 238 行

---

### Q6. 什麼是 All-Reduce？梯度在哪裡進行聚合？
- **30 秒口頭回答：**  
  「All-Reduce 是一種高效的跨設備集體通訊原語（Collective Communication Primitive），它將所有參與設備的數值加總（Reduce），並將加總後的結果同步廣播給所有設備。在我們的程式碼第 232 行，呼叫 `optimizer.apply_gradients(zip(grads, model.trainable_variables))` 時，因為處在 `MirroredStrategy` 範疇內，TensorFlow 內部透過 `strategy.extended` 自動在所有 Replica 間觸發 All-Reduce SUM 操作，將各設備梯度加總後同步更新兩側權重。」
- **對應檔案與函式：** [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py) 之 `step_fn(inputs)`
- **程式碼行號：** 第 232 行

---

### Q7. 分散式訓練中的損失計算為什麼使用 tf.nn.compute_average_loss？
- **30 秒口頭回答：**  
  「這是分散式訓練非常關鍵的數學細節。若直接使用預設的 batch 平均，每個 Replica 只會除以自己的批次大小（32），兩側梯度相加後會膨脹為兩倍，造成嚴重的梯度爆炸或學習率失真。我們將損失函數設為 `Reduction.NONE`，並透過 `tf.nn.compute_average_loss` 將所有樣本的損失除以全域批次大小（64），保證跨設備梯度加總後與單機訓練的數學期望值完全一致。」
- **對應檔案與函式：** [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py) 之 `compute_loss(labels, preds)`
- **程式碼行號：** 第 184～186 行（Reduction.NONE）與第 197～201 行（compute_loss 定義）

---

### Q8. 你如何具體證明兩個 Replica 的權重真的是同步的？
- **30 秒口頭回答：**  
  「我們實作了業界最嚴格的數學級斷言函式 `assert_replicas_synchronized()`。在每個 Epoch 結束時，程式會遍歷模型全部 6 個可訓練權重張量，透過 `strategy.experimental_local_results(var)` 同時取出設備 0 與設備 1 上的實際浮點數值，以 `tf.reduce_all(tf.equal(v0, v1))` 進行逐點比對。只要有任何一個數值存在哪怕 1e-7 的不一致，程式會立即拋出 `RuntimeError` 終止。在全部 71 個 Epoch 中，該斷言 100% 通過。」
- **對應檔案與函式：** [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py) 之 `assert_replicas_synchronized()`
- **程式碼行號：** 第 248～270 行與第 337 行

---

### Q9. 為什麼在終端機能看到 Replica IDs [0, 1] 與 Shard Sizes [32, 32]？
- **30 秒口頭回答：**  
  「這是我們在第 1 個 Epoch 的第 1 個 Batch 特別加入的執行期實測證明（Telemetry Evidence）。透過 `replica_context.replica_id_in_sync_group` 與 `tf.shape(f)[0]` 記錄每個設備的 ID 與樣本數，並以 `strategy.experimental_local_results` 印出。終端機印出 `Replica IDs: [0, 1]` 與 `sharded batch sizes: [32, 32]`，直接證明了兩個獨立的邏輯設備同時處於活躍執行狀態且精準分流。」
- **對應檔案與函式：** [`submission/train_distributed.py`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/train_distributed.py) 之 `main()`
- **程式碼行號：** 第 308～329 行

---

### Q10. 為什麼分散式版本（2.05s/epoch）反而比單機版本（0.39s/epoch）更慢？是否有達到加速？
- **30 秒口頭回答：**  
  「我們在報告中非常誠實地向老師揭露：這個實驗並沒有追求速度上的加速，而是證明分散式架構與同步機制的正確性。因為我們的實體硬體只有一顆 CPU Socket，原本單機版可以完全透過 OpenMP/Eigen 多執行緒在共享快取中高速並行；當虛擬化出兩個邏輯設備時，反而引入了行程內跨執行緒切換、互斥鎖（Mutex Locking）以及每個 Batch 必經的 All-Reduce 通訊開銷。這種通訊成本大於計算節省的現象，在單一 CPU 上是完全符合分散式計算理論的正常結果，展現了真實且透明的系統工程評估。」
- **對應檔案與函式：** [`submission/RESULTS.md`](file:///home/augustine/cloud_computing/midterm_roundabout/submission/RESULTS.md) Section 4 與 [`reports/phase5b_distributed_training.md`](file:///home/augustine/cloud_computing/midterm_roundabout/reports/phase5b_distributed_training.md) Section 6
- **程式碼行號：** `train_distributed.py` 第 15～17 行（Docstring 說明）及 `RESULTS.md` 第 65～80 行
