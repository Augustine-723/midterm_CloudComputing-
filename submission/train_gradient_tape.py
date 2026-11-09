#!/usr/bin/env python3
"""
submission/train_gradient_tape.py
雲端計算期中專題進階加分項目 A：使用 tf.GradientTape 實作客製化訓練與驗證迴圈
Advanced Bonus Deliverable: Custom Training & Validation Loop with tf.GradientTape (Up to 10 Points)

【加分項目符合性說明】
1. 嚴格禁絕 Keras 高階 API：全檔完全不使用 model.fit()、model.evaluate() 與 model.predict()。
2. 低階自動微分：使用 with tf.GradientTape() as tape 上下文管理器記錄前向傳播（Forward Pass）。
3. 顯式梯度計算與更新：透過 tape.gradient() 計算損失對可訓練變數之梯度，並以 optimizer.apply_gradients() 更新權重。
4. 原生資料管線：採用 tf.data.Dataset 進行高效 Shuffle、Batch (64) 與 Prefetch (AUTOTUNE)。
5. 圖編譯加速：在 train_step 與 val_step 上標註 @tf.function 轉換為 TensorFlow 靜態計算圖。
6. 自行實作 EarlyStopping 與 Checkpoint：純 Python 狀態邏輯追蹤 val_loss，耐受度 patience=10，儲存最佳權重。
7. 測試環節說明：本腳本負責模型訓練與驗證；環境端推論測試在 NN_test.py 透過 model(obs, training=False) 執行，
   離線指標評估則由 scripts/evaluate_gradient_tape.py 負責，三者皆完全排除 model.predict()。
"""

import os
import sys
import time
import argparse
import numpy as np

# 抑制 TensorFlow 底層非必要之 C++ 資訊性日誌（僅顯示 Warning 與 Error）
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# 取得目前腳本所在之絕對路徑，確保跨平台與不同執行目錄下的路徑可攜性
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

def resolve_data_dir():
    """
    動態搜尋並解析資料集目錄（Dataset Directory Resolution）。
    
    【設計目的】
    避免將資料集路徑寫死為絕對路徑。當助教在 Docker 容器不同路徑、
    或是 Windows 本機解壓縮執行時，程式能依序在當前目錄、上層目錄中自動定位 train.npz。
    
    Returns:
        str: 包含 train.npz 與 validation.npz 的資料集資料夾絕對路徑。
    """
    candidates = [
        os.path.join(SCRIPT_DIR, "dataset"),
        os.path.join(SCRIPT_DIR, "..", "dataset"),
        os.path.join(os.getcwd(), "dataset"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.npz")):
            return c
    raise FileNotFoundError("Could not locate train.npz in candidate directories: " + str(candidates))

DATA_DIR = resolve_data_dir()
# 預設訓練產出模型路徑（不覆蓋正式交件之 YOURMODEL.h5）
DEFAULT_MODEL_OUT = os.path.join(SCRIPT_DIR, "models", "gradient_tape_model.h5")


def load_datasets():
    """
    載入並預處理訓練集與驗證集資料（Data Loading & Preprocessing）。
    
    【資料維度與形狀說明】
    1. 特徵維度（Input Features）： shape = (-1, 25)
       - 自走車（Ego-vehicle）與周遭 4 輛鄰近車輛之運動學觀測。
       - 共 5 輛車 × 每輛車 5 個屬性 [presence, x, y, vx, vy] = 25 個連續型特徵（float32）。
    2. 標籤形狀（Target Labels）： shape = (-1,)
       - 包含 5 種離散駕駛決策動作的整數索引（int32）：
         0: LANE_LEFT (向左變換車道)
         1: IDLE (保持目前車道與車速)
         2: LANE_RIGHT (向右變換車道)
         3: FASTER (加速行駛)
         4: SLOWER (減速行駛)
    
    Returns:
        tuple: ((x_train, y_train), (x_val, y_val)) 訓練集 (35000 筆) 與驗證集 (5000 筆)。
    """
    train_file = os.path.join(DATA_DIR, "train.npz")
    val_file = os.path.join(DATA_DIR, "validation.npz")

    train_raw = np.load(train_file)
    val_raw = np.load(val_file)

    # 轉型為 float32 並展平為 25 維向量，符合神經網路輸入層規範
    x_train = train_raw["data"].astype(np.float32).reshape(-1, 25)
    y_train = train_raw["label"].astype(np.int32).flatten()

    x_val = val_raw["data"].astype(np.float32).reshape(-1, 25)
    y_val = val_raw["label"].astype(np.int32).flatten()

    return (x_train, y_train), (x_val, y_val)


def build_model():
    """
    構建多層感知機（Multi-Layer Perceptron, MLP）神經網路架構。
    
    【模型結構設計】
    - 輸入層（Input Layer）：25 維連續特徵向量。
    - 隱藏層 1（Dense Layer 1）：128 個神經元，搭配 ReLU 激活函數引入非線性特徵映射。
    - 隱藏層 2（Dense Layer 2）：64 個神經元，搭配 ReLU 激活函數進一步萃取高級特徵。
    - 輸出層（Output Layer）：5 個神經元，搭配 Softmax 激活函數輸出各動作之機率分佈（總和為 1.0）。
    - 總可訓練參數（Trainable Parameters）：11,909 個（與 Baseline 完全一致以達公平比較）。
    
    Returns:
        tf.keras.Sequential: 建立完成但未編譯（Uncompiled）的神經網路模型實例。
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25,), name="input_obs"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(5, activation="softmax", name="output_action")
    ], name="GradientTape_NN")
    return model


def main():
    """
    客製化訓練主程式流程（Main Execution Flow）。
    執行步驟：解析參數 -> 資料管線建置 -> 模型與優化器設定 -> 自訂 Step 函式 -> Epoch 訓練迴圈。
    """
    parser = argparse.ArgumentParser(description="Train with custom tf.GradientTape loop")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_OUT, help="Saved model path")
    args = parser.parse_args()

    # 設定隨機種子（Random Seed），確保權重初始化與資料打亂順序之可重現性
    tf.random.set_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print(" ADVANCED BONUS: CUSTOM TRAINING LOOP WITH TF.GRADIENTTAPE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. 建立高效 tf.data.Dataset 資料管線（Data Pipeline）
    # -------------------------------------------------------------------------
    (x_train, y_train), (x_val, y_val) = load_datasets()
    print(f"Loaded Train: {x_train.shape[0]} samples | Val: {x_val.shape[0]} samples")

    # 訓練資料集管線：
    # 1. from_tensor_slices: 將記憶體中的 NumPy 陣列切片為張量串流。
    # 2. shuffle(buffer_size=10000): 打亂樣本時序關聯，避免車輛軌跡連續幀造成批次偏差。
    # 3. batch(64): 將樣本包裝為 64 筆一組的 mini-batch 進行矩陣並行運算。
    # 4. prefetch(AUTOTUNE): 於 GPU/CPU 訓練當前批次時，非同步預載下一批次，消除 I/O 阻塞瓶頸。
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=10000, seed=42)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 驗證資料集管線（無需 shuffle，保持評估順序一致）
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # -------------------------------------------------------------------------
    # 2. 模型、優化器與損失函數設定（Optimization Setup）
    # -------------------------------------------------------------------------
    model = build_model()
    # Adam 優化器：結合動量法（Momentum）與 RMSprop 自適應學習率，學習率預設 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    # SparseCategoricalCrossentropy 損失函數：適用於類別標籤為純整數索引（0~4）的多分類交叉熵
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # 建立狀態型指標評估器（Stateful Metrics），負責在 Epoch 期間累加樣本損失與準確率
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

    # -------------------------------------------------------------------------
    # 3. 客製化步驟函式（Explicit Step Functions with tf.GradientTape）
    # -------------------------------------------------------------------------
    @tf.function
    def train_step(x_b, y_b):
        """
        單次訓練前向傳播、自動微分與梯度反向傳播（Single Training Step）。
        
        【關鍵技術解析】
        1. @tf.function: 透過 Python 追蹤（Tracing）與 AutoGraph 將函式轉換為 TensorFlow 靜態計算圖（FunctionGraph），
           減少 Python 解譯器反覆執行之直譯開銷（註：此為 Graph Tracing 機制，非 XLA JIT 編譯；除非顯式指定 jit_compile=True）。
        2. with tf.GradientTape() as tape: 核心上下文管理器，在計算圖中動態追蹤所有涉及可訓練變數的運算。
        3. model(x_b, training=True): 設定為訓練模式（確保日後若引入 Dropout 或 BatchNorm 能正確啟用）。
        4. loss_fn(y_b, preds): 計算該批次的平均交叉熵損失，衡量模型預測分佈與實際真實標籤之差距。
        5. tape.gradient(loss, model.trainable_variables): 執行逆向自動微分（Reverse-mode Autodiff / Vector-Jacobian Product, VJP），
           依微積分連鎖律從純量損失逆向追蹤中間運算張量，高效率精確求解損失對所有權重與偏差張量之一階偏微分梯度向量 ∇_W (Loss)。
        6. optimizer.apply_gradients(): 將梯度應用於 Adam 更新公式，顯式更新網路權重。
        """
        with tf.GradientTape() as tape:
            preds = model(x_b, training=True)
            loss = loss_fn(y_b, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_metric.update_state(loss)
        train_acc_metric.update_state(y_b, preds)
        return loss

    @tf.function
    def val_step(x_b, y_b):
        """
        單次驗證前向傳播（Single Validation Step）。
        
        【關鍵技術解析】
        1. model(x_b, training=False): 設定為推論/評估模式。
        2. 無須 GradientTape: 不記錄計算軌跡，節省反向傳播之記憶體與運算開銷。
        3. 不更新權重：嚴格保持權重不變，以未參與訓練之獨立驗證集檢驗模型的真實泛化能力（Generalization）。
        """
        preds = model(x_b, training=False)
        loss = loss_fn(y_b, preds)
        val_loss_metric.update_state(loss)
        val_acc_metric.update_state(y_b, preds)
        return loss

    # -------------------------------------------------------------------------
    # 4. 自訂訓練迴圈與 EarlyStopping 機制（Custom Epoch Loop & Checkpointing）
    # -------------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_cnt = 0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # 每個 Epoch 開始時，重置評估指標累加器
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()

        # 遍歷訓練資料集中的所有 mini-batches 執行梯度更新
        for xb, yb in train_ds:
            train_step(xb, yb)

        # 遍歷驗證資料集中的所有 mini-batches 進行推論評估
        for xb, yb in val_ds:
            val_step(xb, yb)

        dur = time.time() - t0
        t_loss = float(train_loss_metric.result())
        t_acc = float(train_acc_metric.result()) * 100.0
        v_loss = float(val_loss_metric.result())
        v_acc = float(val_acc_metric.result()) * 100.0

        # 手工實作之早停與最優權重保存（EarlyStopping & ModelCheckpoint Logic）：
        # 以 validation loss 作為監控指標；若損失改善超過門檻（1e-4），則保存最優模型權重並重置計數器；
        # 若連續 10 個 Epoch 均未改善，則提前終止訓練，防止神經網路對訓練資料過擬合（Overfitting）。
        if v_loss < best_val_loss - 1e-4:
            best_val_loss = v_loss
            patience_cnt = 0
            model.save(args.output)
            tag = f"[Saved -> val_loss: {v_loss:.5f}]"
        else:
            patience_cnt += 1
            tag = f"[Patience: {patience_cnt}/10]"

        print(f"Epoch {epoch:02d}/{args.epochs:02d} ({dur:.2f}s) — loss: {t_loss:.4f} - acc: {t_acc:.2f}% | val_loss: {v_loss:.4f} - val_acc: {v_acc:.2f}% {tag}")

        if patience_cnt >= 10:
            print(f"EarlyStopping triggered. Best val_loss: {best_val_loss:.5f}")
            break

    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()
