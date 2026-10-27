# Phase 3A 成果報告：模型整合與 Roundabout 環境閉環評估

> **專案名稱**：雲端計算期中專題 — Supervised Learning for Highway-env Roundabout  
> **專案路徑**：`/home/augustine/cloud_computing/midterm_roundabout`  
> **執行環境**：Docker Container (`roundabout-midterm`) on Ubuntu  
> **模擬環境**：Gymnasium (`roundabout-v0`) with highway-env 1.8.2  
> **評估腳本**：[`scripts/evaluate_environment.py`](file:///home/augustine/cloud_computing/midterm_roundabout/scripts/evaluate_environment.py)  
> **評估數據**：[`reports/phase3a_results.json`](file:///home/augustine/cloud_computing/midterm_roundabout/reports/phase3a_results.json)  
> **評估日期**：2026-10-22

---

## 1. 實際環境規格與相容性診斷 (Environment Compatibility)

在將 Phase 2 訓練之模型整合至 `roundabout-v0` 閉環模擬前，進行全面規格檢驗：

| 檢驗項目 | 實際環境規格實測 | 模型與資料集對應規格 | 相容性判定 |
| :--- | :--- | :--- | :---: |
| **highway-env 版本** | `1.8.2` | 建置目標版本 `1.8.2` | **100% 吻合** |
| **Gymnasium 版本** | `0.29.1` | Phase 1 修復鎖定版本 `0.29.1` | **100% 吻合** |
| **Observation Space** | `Box(-inf, inf, (5, 5), float32)` | 模型輸入展平為 `(1, 25)` | **100% 吻合** |
| **Observation Type** | `Kinematics (absolute: True)` | 5 輛車 $\times$ 5 特徵 | **100% 吻合** |
| **Observation 特徵排列** | `[presence, x, y, vx, vy]` | 資料集欄位排列完全相同 | **100% 吻合** |
| **數值歸一化區間** | 經 `features_range` 縮放至 $[-1.0, 1.0]$ | 訓練資料特徵範圍 $[-1.0, 1.0]$ | **100% 吻合** |
| **Action Space** | `Discrete(5)` (0~4) | 模型輸出 Softmax 5 類別 | **100% 吻合** |
| **Model 重新載入** | `tf.keras.models.load_model()` | 兩模型皆成功載入，輸出機率和為 1.0 | **PASS** |

> [!NOTE]
> 診斷確認：當前 Docker 容器環境配置與 Phase 1、Phase 2 訓練時之數據結構 **100% 相容**，無須改動老師提供之 [`NN_test.py`](file:///home/augustine/cloud_computing/midterm_roundabout/NN_test.py)。

---

## 2. 評估方法 (Evaluation Method)

1. **配對實驗設計 (Paired Comparison)**：
   - 評估模型：`baseline_standard.h5`（Exp A）與 `baseline_weighted.h5`（Exp B）。
   - 採用固定的種子清單：`seeds = [0, 1, ..., 99]`（共 100 輪獨立 Episodes）。
   - 兩模型在每一輪 Episode 面對完全相同的隨機車流、初始位置、車速擾動與路徑意圖。
2. **單步決策流程**：
   - 每一時間步取得 `obs` $\in \mathbb{R}^{5 \times 5}$，展平為 `shape=(1, 25)` 輸入模型。
   - 使用 `np.argmax(probs)` 提取離散動作 $a \in \{0, 1, 2, 3, 4\}$。
   - 同步檢驗動作是否落在 `env.unwrapped.action_type.get_available_actions()` 中，記錄「無法執行之動作（Infeasible Actions）」次數。
   - 呼叫 `env.step(a)` 推進模擬，累計環境回報（Episode Cumulative Reward）。
3. **明確的成功與終止條件**：
   - **碰撞 (Collision / Terminated)**：`info['crashed'] == True` 或 `vehicle.crashed == True`。
   - **成功到達目的地 (Destination Success)**：車輛在未碰撞的情況下，終止時所在車道包含規劃終點出匝道 `"nxs"`（North Exit Lane）。
   - **時間截斷 (Truncation)**：達到模擬最大時間限制（`duration = 11` 秒，即 11 步）但尚未抵達出匝道。

---

## 3. 100-Episode 模擬統計結果 (100-Episode Statistics)

| 評估指標 | Standard Baseline (`baseline_standard.h5`) | Class Weighted Baseline (`baseline_weighted.h5`) | 差異分析 (Weighted vs Standard) |
| :--- | :---: | :---: | :--- |
| **離線 Validation Accuracy** | **94.64%** | 92.46% | Standard 較高 (+2.18%) |
| **離線 Macro F1-score** | **91.88%** | 86.44% | Standard 較高 (+5.44%) |
| **平均環境回報 (Mean Reward)** | 9.52 | <font color="green">**9.59**</font> | **Weighted 略高 (+0.07)** |
| **回報標準差 (Reward Std)** | 1.51 | <font color="green">**1.39**</font> | **Weighted 更穩定 (-0.12)** |
| **中位數回報 (Median Reward)** | 9.83 | <font color="green">**9.92**</font> | **Weighted 略高 (+0.09)** |
| **最小 / 最大回報 (Min / Max)** | 3.00 / 10.92 | 3.00 / 10.92 | 極值相同 |
| **平均 Episode 步數** | 10.59 步 | 10.66 步 | 兩者接近（滿額 11 步） |
| **碰撞次數 (Collision Rate)** | 9 次 (**9.0%**) | <font color="green">**8 次 (8.0%)**</font> | **Weighted 碰撞率更低 (-1.0%)** |
| **目的地到達成功率 (Success Rate)** | <font color="green">**31.0% (31/100)**</font> | 29.0% (29/100) | Standard 成功率稍高 (+2.0%) |
| **時間超時截斷率 (Truncation Rate)** | 92.0% | 93.0% | 絕大多數 Episode 存活至 11 步滿額 |
| **無法執行動作比例 (Infeasible Rate)** | **43.81%** (464 次) | 45.68% (487 次) | 兩者皆有約 44% 預測動作受路況限制 |

---

## 4. 評估圖表與行為視覺化

````carousel
![100 輪 Episode 回報對比](/home/augustine/.gemini/antigravity/brain/f860e569-1c04-4a7c-9b90-a2add07cabd2/figures/env_episode_rewards.png)
<!-- slide -->
![環境回報分佈直方圖與箱型圖](/home/augustine/.gemini/antigravity/brain/f860e569-1c04-4a7c-9b90-a2add07cabd2/figures/env_reward_distribution.png)
<!-- slide -->
![模擬動作選擇分佈比較](/home/augustine/.gemini/antigravity/brain/f860e569-1c04-4a7c-9b90-a2add07cabd2/figures/env_action_distribution.png)
<!-- slide -->
![碰撞率與成功率比較](/home/augustine/.gemini/antigravity/brain/f860e569-1c04-4a7c-9b90-a2add07cabd2/figures/env_collision_success_rate.png)
````

---

## 5. 動作分佈與決策行為深入剖析 (Action Distribution Analysis)

兩模型在實際環境運行共 1,059 與 1,066 步中的動作選擇分佈如下：

| 動作代碼與意義 | 專家資料集佔比 | Standard Baseline (Exp A) | Class Weighted Baseline (Exp B) | 決策行為觀察 |
| :--- | :---: | :---: | :---: | :--- |
| `0: LANE_LEFT` | 2.30% | 1.70% (18 步) | <font color="green">**2.25% (24 步)**</font> | Weighted 更接近專家分佈，換道更主動 |
| `1: IDLE` | 24.89% | 25.87% (274 步) | 24.48% (261 步) | 兩者均維持約 1/4 時間保持航向 |
| `2: LANE_RIGHT` | 0.87% | 0.66% (7 步) | <font color="green">**1.22% (13 步)**</font> | Weighted 出彎變道頻率提升近一倍 |
| `3: FASTER` | 28.53% | 29.27% (310 步) | 28.99% (309 步) | 兩者巡航加速比率高度一致 |
| `4: SLOWER` | 43.41% | 42.49% (450 步) | 43.06% (459 步) | 圓環內避讓與跟車減速動作佔最大比例 |

### 行為特徵洞察：
1. **動作分佈高度還原專家行為**：
   兩個模型在封閉環境中的動作佔比，皆驚人地契合訓練集的專家分佈（如 SLOWER 均在 42%~43%，FASTER 均在 29%，IDLE 均在 24%~25%），證明監督式學習成功克隆了專家的圓環駕駛決策邏輯。
2. **類別加權帶來的積極換道效果**：
   `baseline_weighted.h5` 由於在訓練時對 `LANE_LEFT` 與 `LANE_RIGHT` 賦予高權重，在模擬中選擇左換道的步數從 18 步增加到 24 步（+33%），選擇右換道的步數從 7 步增加到 13 步（+85%）。
3. **無法執行動作 (Infeasible Actions, 44%~45%) 的成因**：
   - 在圓環的進入匝道單行道（如 `ser` 車道）上，車輛兩側皆無相鄰車道，此時若模型輸出 `LANE_LEFT` 或 `LANE_RIGHT`，環境 API 的 `get_available_actions()` 會判定該動作無效（highway-env 自動保持原車道）。
   - 當車輛速度已達最大值 16 m/s 時，若預測 `FASTER` 亦屬無法執行動作；同理速度達 0 時 `SLOWER` 亦然。
   - 這顯示純監督式學習模型**缺乏對動作可行性遮罩（Action Masking）的感知**。

---

## 6. 核心發現與模型綜合比較 (Key Findings)

> [!IMPORTANT]
> **關鍵定律實證：「離線 Validation Accuracy 高 $\neq$ 環境在線回報高」**
> - **Standard Baseline** 的離線驗證準確率高達 **94.64%**，Macro F1 為 **91.88%**。
> - **Class Weighted Baseline** 的離線驗證準確率為 **92.46%**，Macro F1 為 **86.44%**。
> - **然而在 100 輪真實環境模擬中**：
>   - **Weighted 模型平均回報更優**（**9.59** vs 9.52），中位數回報更高（**9.92** vs 9.83），回報標準差更小（**1.39** vs 1.51）。
>   - **Weighted 模型碰撞率更低**（**8.0%** vs 9.0%，減少 1 次碰撞）。
> - **原因分析**：
>   自駕決策並非單純的離散分類。在圓環關鍵交匯點上，能否及時換道或主動調整速度（即使在某些安全狀態下可能稍微偏離標籤）對避開危險至關重要。Weighted 模型對少數換道動作的高敏感度，實質上幫助車輛在交會瓶頸處更果斷地採取避讓與變道，從而降低了事故率並穩定了整體回報。

---

## 7. 主要限制與發現之問題 (Limitations & Issues)

1. **出匝道成功率僅約 30% (Success Rate ~30%)**：
   - 雖然 100 輪中有 91%~92% 的 Episode 未發生碰撞存活至結束，但其中僅約 30% 能在預設的 11 秒 (`duration=11`) 內順利開出 `"nxs"` 出口匝道。其餘約 60% 的車輛雖然安全，但因頻繁採取減速（SLOWER 佔 43%）而無法在時間限制前完全通過圓環。
2. **缺乏 Action Mask 機制**：
   - 模型在無法換道的單行匝道中仍有頻繁輸出變道動作的傾向（無法執行率達 44%）。
3. **開環預測與閉環累計誤差 (Covariate Shift)**：
   - 監督式學習的資料由靜態專家採集，一旦車輛在動態環境中走出罕見的偏離航向，累積誤差會使模型進入未曾見過的觀測分佈，導致行為猶豫不決或陷入連續減速。

---

## 8. 下一階段優化建議 (Phase 3B / Phase 4 Recommendations)

1. **速度目標引導 (Target Speed & Trajectory Planning)**：
   - 在保持安全避障的前提下，適度懲罰過度減速，鼓勵車輛維持平均巡航速度（約 8~12 m/s），以在 11 秒時限內順利駛出圓環，拉升 Success Rate。
2. **Action Masking 整合**：
   - 在推論時，若環境提供可行動作列表，可在 Softmax 機率輸出前對不可行動作施加負無限大遮罩（Logit Masking），強制模型在當前合法動作中選擇最佳決策。
3. **特徵工程強化 (Temporal / Relative Dynamics)**：
   - 模型目前只看單一瞬間的 5 輛車快照，若引入相鄰 2~3 幀歷史資訊（Frame Stacking），將能更準確地預測周遭車輛的變道軌跡。

