# Phase 3B 成果報告：超參數優化實驗與模型環境行為評估

> **專案名稱**：雲端計算期中專題 — Supervised Learning for Highway-env Roundabout  
> **專案路徑**：`/home/augustine/cloud_computing/midterm_roundabout`  
> **執行環境**：Docker Container (`roundabout-midterm`) on Ubuntu  
> **訓練腳本**：[`scripts/train_optimized.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/train_optimized.py)  
> **評估腳本**：[`scripts/evaluate_optimization.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/evaluate_optimization.py)  
> **完整數據**：[`reports/phase3b_results.json`](file:///home/augustine/cloud_computing/midterm_roundabout/reports/phase3b_results.json)  
> **報告日期**：2026-10-22

---

## 1. 超參數實驗設定 (Hyperparameter Settings)

本階段以「提升 Validation Accuracy 為首要目標，同時檢驗模型在 `roundabout-v0` 中的在線行車行為」展開三組對比實驗，所有模型皆採用相同前處理、隨機種子（`seed=42`）與回呼條件（EarlyStopping: patience=10, ModelCheckpoint）：

| 實驗組別 | 模型架構 (Architecture) | 參數量 (Parameters) | 學習率 ($\eta$) | 批次大小 / 最大輪數 | 儲存路徑 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp A: Baseline (Control)** | `25 -> 128 -> 64 -> 5` | 11,909 | 0.001 | 64 / 100 | `models/baseline_standard.h5` |
| **Exp B: Wider Network** | `25 -> 256 -> 128 -> 5` | 40,197 (+238%) | 0.001 | 64 / 100 | `models/optimized_wider.h5` |
| **Exp C: Lower Learning Rate**| `25 -> 128 -> 64 -> 5` | 11,909 | 0.0003 (-70%) | 64 / 100 | `models/optimized_low_lr.h5` |

---

## 2. 訓練收斂過程與曲線 (Training Results & Curves)

````carousel
![超參數收斂曲線對比](/home/augustine/.gemini/antigravity/brain/f860e569-1c04-4a7c-9b90-a2add07cabd2/figures/opt_training_curves.png)
<!-- slide -->
![離線 Validation Accuracy 與 Macro F1 對比](/home/augustine/.gemini/antigravity/brain/f860e569-1c04-4a7c-9b90-a2add07cabd2/figures/opt_hyperparameter_comparison.png)
<!-- slide -->
![三模型混淆矩陣對照](/home/augustine/.gemini/antigravity/brain/f860e569-1c04-4a7c-9b90-a2add07cabd2/figures/opt_confusion_matrices.png)
<!-- slide -->
![各類別 Recall 召回率比較](/home/augustine/.gemini/antigravity/brain/f860e569-1c04-4a7c-9b90-a2add07cabd2/figures/opt_per_class_recall.png)
<!-- slide -->
![在線環境評估（Control vs Wider）](/home/augustine/.gemini/antigravity/brain/f860e569-1c04-4a7c-9b90-a2add07cabd2/figures/opt_env_comparison.png)
````

### 訓練特徵觀察：
1. **Wider Network (Exp B)**：
   - 參數量擴增至 40,197，訓練速度依然迅速（45.6 秒，每 Epoch 約 1 秒）。
   - 在第 38 輪達到最佳驗證損失 `0.1508`（驗證準確率 94.28%），第 48 輪觸發 Early Stopping。
   - 觀察曲線可見訓練損失持續下降至 0.113，但驗證損失在 0.150 附近平原停滯，呈現**輕微過擬合（Slight Overfitting）**跡象。
2. **Lower Learning Rate (Exp C)**：
   - 學習率調降為 0.0003，收斂速度顯著平緩，直至第 97 輪才達到最低驗證損失 `0.1722`（驗證準確率 93.84%）。
   - 100 輪內並未遭遇顯著過擬合，但由於學習率過小，在有限輪數內未收斂至更深谷底。

---

## 3. 離線驗證集評估對照 (Offline Validation Comparison)

在相同之 `dataset/validation.npz`（5,000 筆樣本）上進行嚴格盲測：

| 評估指標 | Exp A: Baseline (Control) | Exp B: Wider Network | Exp C: Lower LR | 最佳表現者 |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Accuracy (主指標)**| <font color="green">**94.64%**</font> | 94.28% (-0.36%) | 93.84% (-0.80%) | **Exp A (Control)** |
| **Validation Loss** | <font color="green">**0.1498**</font> | 0.1508 | 0.1722 | **Exp A (Control)** |
| **Macro Precision** | <font color="green">**91.03%**</font> | 90.69% | 88.37% | **Exp A (Control)** |
| **Macro Recall** | **92.92%** | <font color="green">**93.43%**</font> | 89.92% | **Exp B (Wider)** |
| **Macro F1-score** | <font color="green">**91.88%**</font> | 91.95% | 89.04% | **Exp B / Exp A 相當** |
| **Weighted F1-score** | <font color="green">**94.64%**</font> | 94.29% | 93.87% | **Exp A (Control)** |
| **總錯誤筆數 (Total Errors)** | <font color="green">**268 筆**</font> | 286 筆 (+18 筆) | 308 筆 (+40 筆) | **Exp A (最少錯誤)** |

---

## 4. 分類錯誤剖析 (Classification Error Analysis)

針對 Task B 發現之瓶頸，深度比對三模型之錯誤結構：

| 混淆指標 / 類別 Recall | Exp A: Baseline (Control) | Exp B: Wider Network | Exp C: Lower LR | 改善/惡化判定 |
| :--- | :---: | :---: | :---: | :---: |
| **FASTER $\rightarrow$ SLOWER 誤判** | **53 筆 (3.72%)** | 66 筆 (4.63%) | 67 筆 (4.71%) | Exp A 較佳 |
| **SLOWER $\rightarrow$ FASTER 誤判** | 53 筆 (2.49%) | <font color="green">**42 筆 (1.97%)**</font> | <font color="green">**33 筆 (1.55%)**</font> | **Exp B & C 顯著改善** |
| **雙向互斥混淆總筆數 (F $\leftrightarrow$ S)**| **106 筆** | 108 筆 | **100 筆** | Exp C 總混淆最少 |
| **Class 0 (LANE_LEFT) Recall** | 88.06% (118/134) | <font color="green">**89.55% (120/134)**</font> | 84.33% (113/134) | **Exp B 提升 +1.49%** |
| **Class 2 (LANE_RIGHT) Recall** | **92.50% (37/40)** | **92.50% (37/40)** | 92.50% (37/40) | 三者完全持平 |

### 剖析結論：
1. **加寬網路 (Wider Network)** 確實在特定難題上展現了更強的非線性擬合能力：
   - `SLOWER -> FASTER` 的嚴重逆向誤判從 53 筆銳減至 42 筆（降幅達 21%）。
   - `LANE_LEFT` 召回率從 88.06% 微升至 89.55%（多捕捉 2 筆換道）。
   - 然而，由於網路容量較大，在多數類別的邊界上增加了其他細微擾動，整體錯誤多出 18 筆，使得總體 Accuracy 微跌 0.36%。
2. **較小學習率 (Lower LR)**：
   - 雖然將 `SLOWER -> FASTER` 壓低至 33 筆，但因整體收斂速度慢，對少數類別 `LANE_LEFT` 的召回率跌至 84.33%，總體表現未達預期。

---

## 5. 在線環境模擬評估 (Task E: Environment Results)

依規範，選出新模型中 Validation Accuracy 最高的 **Wider Network (Exp B, 94.28%)**，與 **Baseline Standard (Control)** 在 `roundabout-v0` 進行 100 輪種子配對模擬（`seeds = 0 ~ 99`）：

| 環境評估指標 | Baseline Standard (Control) | Wider Network (Exp B) | 差異比較 |
| :--- | :---: | :---: | :---: |
| **平均環境回報 (Mean Reward)** | 9.52 | **9.53** | 差異僅 `+0.015` |
| **回報標準差 (Reward Std)** | 1.51 | **1.41** | Wider 波動略小 |
| **中位數回報 (Median Reward)** | 9.83 | 9.83 | 完全相同 |
| **碰撞率 (Collision Rate)** | 9.0% (9 次) | 9.0% (9 次) | 完全相同 (9 vs 9) |
| **目的地到達率 (Arrival Rate)** | 31.0% (31 次) | 31.0% (31 次) | 完全相同 (31 vs 31) |
| **平均實際車速 (Actual Speed)** | 6.76 m/s | 6.73 m/s | 差異僅 0.03 m/s |
| **平均目標車速 (Target Speed)** | 6.83 m/s | 6.79 m/s | 差異僅 0.04 m/s |
| **停滯/爬行時間佔比 (Speed < 1m/s)** | 30.97% | 31.10% | 兩者皆有約 31% 時間受阻停滯 |
| **無法執行換道比例 (Infeasible Lane)**| 0.66% (7 步) | 0.85% (9 步) | 兩者皆極低，道路邊界感知良好 |
| **速度限制下無效加減速 (Speed Saturation)**| 43.15% (457 步) | 42.88% (455 步) | 兩者速度飽和率完全相當 |
| **總無效動作比例 (Total Invalid)** | 43.81% | 43.73% | 兩者無統計差異 |

---

## 6. 配對 Bootstrap 信賴區間統計分析 (Statistical Comparison)

對 100 輪配對 Episode 的 Reward 差異與 Collision 差異進行 10,000 次 Bootstrap 抽樣分析：

1. **Mean Reward Difference ($\text{Wider} - \text{Control}$)**：
   - 樣本均值差：`+0.0146`
   - **95% Bootstrap 信賴區間**：`[-0.2688, +0.2988]`
   - **判定**：區間對稱且包含 0，未達統計顯著水準。
2. **Collision Rate Difference ($\text{Wider} - \text{Control}$)**：
   - 樣本碰撞差：`0.00%`（皆為 9 次事故）
   - **95% Bootstrap 信賴區間**：`[-6.00%, +6.00%]`
   - **判定**：兩模型在安全性上無任何顯著差異。

---

## 7. 最佳模型決策建議 (Best Model Recommendation)

綜合離線驗證集指標與在線環境表現：

> [!IMPORTANT]
> **最終推薦主模型：維持 `models/baseline_standard.h5`**
> 
> **決策理由**：
> 1. **Accuracy 最高**：Baseline Standard 在驗證集取得 **94.64%**，錯誤筆數最少（268 筆，比 Wider 少 18 筆，比 Lower LR 少 40 筆）。
> 2. **在線表現等價**：在 `roundabout-v0` 100 輪真實模擬中，Wider Network 與 Baseline 的 Reward（9.53 vs 9.52）、碰撞率（9.0% vs 9.0%）、出匝道到達率（31.0% vs 31.0%）在統計上完全等價。
> 3. **參數量精簡**：Baseline Standard 僅需 11,909 個參數（46 KB），推論極其迅速且不易過擬合；Wider Network 參數量達 40,197，未帶來足額泛化回報。

---

## 8. 主要限制與下一階段優化建議 (Limitations & Next Steps)

1. **單純擴寬隱藏層遭遇邊際效益遞減 (Diminishing Returns)**：
   - 單純將神經元增加至 256-128，無法突破特徵資訊瓶頸。這是因為特徵 0（主車存在）為無效特徵，且純 MLP 缺乏對 5 輛車之間空間幾何關係的歸納偏置。
2. **低到達率 (Arrival Rate 31%) 的根本原因已查明**：
   - 本次實驗實測發現，車輛在圓環中**有高達 31% 的時間處於實際速度 $< 1$ m/s 的停滯狀態**。
   - 頻繁的 `SLOWER` 輸出導致車輛在遭遇前方車流時過度減速並在匝道等待，而在 11 秒限時到達時尚未能行駛足夠距離到達北出口。
3. **建議下一步改進方向 (Phase 4)**：
   - **架構升級 (Residual-MLP / Feature Interaction)**：引入殘差連線或車輛相對距離計算層，提高前車距與相對車速的表徵精度。
   - **速度目標引導**：適度加入維持目標速度（8~12 m/s）的微調，平衡避障與前進動能，拉升出匝道到達率。

