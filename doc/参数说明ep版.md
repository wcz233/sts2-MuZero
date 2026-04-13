# sts2-MuZero 参数说明（ep版）

## 1. 适用范围

本文档只说明当前 **episode-end training** 版本真正生效的主参数。

当前训练流程是：

1. 环境逐步交互，收集本局所有 `Transition`
2. 本局结束后统一做 episode 级结算
3. 再把整局样本写入 replay
4. 按 `--updates-per-episode` 做一次或多次训练更新
5. checkpoint 同时保存当前模型状态和 `episode_index`

因此，旧时代的 step 版参数不应再作为主参数使用。

## 2. 先看最重要的结论

如果只记三件事：

1. `--updates-per-episode` 现在是训练频率主开关，不要再用 `--updates-per-step` 作为主配置。
2. `--floor-advance-weight` 不能再沿用旧版的大值，因为现在楼层奖励已经按 `0.5 * n * (1.1^n)` 递增。
3. `--turn-skip-unspent-penalty` 建议直接给显式值，不建议继续依赖 `--turn-skip-unspent-multiplier 100` 这种放大式配置。

## 3. 参数如何映射到“模型特征”

可以把参数分成 6 类：

1. 搜索强度
2. 探索强度
3. 学习稳定性
4. reward 偏好
5. 行为合法性与 UI 兼容
6. 日志与恢复

下面按这个结构解释。

## 4. 搜索与行动参数

### `--simulations`

- 作用：控制普通状态下 MuZero 根节点搜索次数。
- 影响模型特征：影响“每一步下决策前愿意想多久”。
- 调大后：
  - 决策更稳定
  - 行为更像 exploitation
  - 速度更慢
- 调小后：
  - 搜索更快
  - 数据更杂
  - 容易产生短视动作

建议：

- 日常训练：`32 ~ 48`
- 收敛/复训：`48 ~ 64`
- 快速冒烟：`12 ~ 20`

### `--combat-selection-simulations`

- 作用：控制 `hand_select` 这类“选牌界面”的专用搜索次数上限。
- 影响模型特征：影响复杂选择界面的稳定性和耗时。
- 调大后：
  - 选牌更稳
  - 但在大量 overlay 场景里更慢

建议：

- 日常训练：`8 ~ 12`
- 快速调试：`4 ~ 8`

### `--combat-selection-search-time-budget-seconds`

- 作用：给 `hand_select` 额外加一个墙钟时间预算。
- 影响模型特征：限制某些慢界面把训练卡死。

建议：

- 日常训练：`0.02 ~ 0.05`
- 如果还嫌慢：继续降到 `0.02`
- 如果你更在意稳定：可以升到 `0.06 ~ 0.08`

### `--temperature`

- 作用：控制动作采样温度。
- 影响模型特征：直接决定“策略更随机还是更保守”。
- 高温：
  - 更探索
  - 日志更丰富
  - 更容易出现噪声行为
- 低温：
  - 更稳定
  - 更适合复训/收敛
  - 容易陷入当前策略局部最优

建议：

- 采集训练数据：`0.45 ~ 0.70`
- 稳定主训练：`0.35 ~ 0.50`
- 收敛/复训：`0.15 ~ 0.30`

### `--no-root-noise`

- 作用：关闭根节点探索噪声。
- 影响模型特征：让搜索更 deterministic。

建议：

- 日常训练：默认不要开
- 收敛复训、对比实验、稳定复盘：可以开

## 5. 学习与 replay 参数

### `--learning-rate`

- 作用：优化器学习率。
- 影响模型特征：决定“每一轮参数更新迈多大步”。
- 过大：
  - loss 抖动大
  - 容易把刚学到的东西冲掉
- 过小：
  - 学得太慢
  - 实机训练回报不明显

建议：

- 稳定训练：`1e-4 ~ 2e-4`
- 当前版本推荐起点：`1.5e-4`
- 如果出现明显震荡：先降到 `1e-4`

### `--discount`

- 作用：价值回传折扣因子。
- 影响模型特征：决定模型更看重长期收益还是短期收益。

建议：

- 默认 `0.997` 是合理起点
- 不建议频繁改
- 若想更短视：`0.99 ~ 0.995`
- 若更看长期：保持 `0.997`

### `--updates-per-episode`

- 作用：每个 episode 结束后做多少次 replay SGD 更新。
- 影响模型特征：决定“每打完一局学多少”。
- 调大后：
  - 学得更快
  - 更容易过拟合最近数据
- 调小后：
  - 更稳
  - 但学习速度变慢

建议：

- 日常训练：`4 ~ 10`
- 快速调试：`1 ~ 3`
- 收敛复训：`8 ~ 16`

### `--replay-capacity`

- 作用：replay buffer 大小。
- 影响模型特征：决定“模型记忆窗口有多长”。
- 大 replay：
  - 更稳
  - 不容易忘掉旧经验
  - 学得稍慢
- 小 replay：
  - 更快适应新策略
  - 但更容易遗忘、波动

建议：

- 日常训练：`16000 ~ 30000`
- 如果你只跑很短实验：`8000 ~ 12000`

### `--warmup-samples`

- 作用：replay 里积累到多少样本后才开始真正训练。
- 影响模型特征：决定前期是不是先积累数据再学。

建议：

- 日常训练：`1000 ~ 2000`
- 快速验证：`200 ~ 500`

### `--checkpoint-path`

- 作用：checkpoint 路径。
- 影响模型特征：不改模型行为，但决定训练状态保存位置。

### `--checkpoint-interval`

- 作用：按 step 周期保存 checkpoint。
- 影响模型特征：不直接影响策略，但影响断点恢复粒度。

建议：

- 日常训练：`200 ~ 500`
- 快速调试：`50 ~ 100`

### `--resume`

- 作用：从 checkpoint 恢复模型和 `episode_index`。
- 影响模型特征：保证训练连续性，不会把 ep 编号重新从 1 开始。

## 6. reward 参数

这一组最重要，因为它们直接定义“模型学会偏好什么”。

### `--floor-advance-weight`

- 作用：层数推进奖励基权重。
- 影响模型特征：鼓励模型活得更久、走得更深。
- 当前版本注意：
  - 实际奖励不是线性的
  - 每层奖励为 `floor_advance_weight * 0.5 * n * (1.1^n)`

这意味着旧版 `5.0` 已经非常大。

建议：

- 当前版本推荐起点：`0.08 ~ 0.15`
- 不建议再用 `1.0+`
- 更不建议继续用旧版 `5.0`

### `--act-advance-weight`

- 作用：进入下一幕的奖励。
- 影响模型特征：鼓励完整推进 run，而不是只吃单层 shaping。

建议：

- `4 ~ 10`

### `--hp-delta-weight`

- 作用：玩家 HP 变化权重。
- 影响模型特征：模型是否珍惜血量。

建议：

- `4 ~ 6`
- 太低会变莽
- 太高会过于保守

### `--gold-delta-weight` / `--gold-delta-scale`

- 作用：金币变化奖励。
- 影响模型特征：模型是否重视经济路线、商店价值。

建议：

- `gold-delta-weight`: `0.2 ~ 0.4`
- `gold-delta-scale`: 保持 `25`

### `--enemy-hp-delta-weight`

- 作用：敌方掉血奖励。
- 影响模型特征：鼓励模型积极推进战斗。

建议：

- `6 ~ 8`

### `--turn-end-weight`

- 作用：玩家回合结束的即时奖励。
- 影响模型特征：鼓励“正常结束回合”。

当前版本建议：

- 默认用 `0.0`
- 因为未用能量惩罚已经足够重要
- 不建议再单独给它太多正奖励

### `--turn-skip-unspent-penalty`

- 作用：显式定义“有可打牌但结束回合且剩能量”的惩罚。
- 影响模型特征：控制模型是否乱空过回合。

这是当前版本的推荐主开关。

建议：

- `4 ~ 8`
- 当前推荐起点：`6.0`

### `--turn-skip-unspent-multiplier`

- 作用：当未显式设置 penalty 时，按“全局最大正奖励”推导未用能量惩罚。
- 影响模型特征：会把 skip penalty 绑定到其他 reward 量级。

不推荐作为主配置使用，原因是：

1. 当前楼层奖励已指数递增
2. 全局最大正奖励会被放大
3. 会把每点未用能量惩罚一起拉爆

建议：

- 日常训练不要依赖它
- 直接设 `--turn-skip-unspent-penalty`

### `--combat-end-weight` / `--combat-defeat-weight`

- 作用：战斗结束奖励 / 战斗死亡惩罚。
- 影响模型特征：鼓励赢下战斗，并拉开生死差异。

建议：

- `combat-end-weight`: `5 ~ 8`
- `combat-defeat-weight`: `-8 ~ -12`

### `--act-end-weight`

- 作用：过幕奖励。
- 影响模型特征：强化长期推进。

建议：

- `1 ~ 3`

### `--run-victory-weight` / `--run-defeat-weight`

- 作用：整局胜负终局奖励。
- 影响模型特征：影响整局层面的价值偏好。

建议：

- `run-victory-weight`: `10 ~ 15`
- `run-defeat-weight`: `-15 ~ -20`

### `--combat-tactical-shaping`

- 作用：战斗局部势函数 shaping。
- 影响模型特征：帮助模型学会更合理的战斗中间步骤，而不是只看最终输赢。

建议：

- `0.25 ~ 0.35`
- 当前推荐起点：`0.30`

### `--end-turn-slack-penalty`

- 作用：额外的松弛型回合结束惩罚。
- 影响模型特征：也会抑制乱空过回合。

当前版本建议：

- 如果你已经显式设置了 `--turn-skip-unspent-penalty`，建议设为 `0.0`
- 避免双重惩罚

### `--episode-settlement-weight`

- 作用：整个 episode 结束后的 retroactive credit 权重。
- 影响模型特征：让模型更重视“整局走向”，而不是只盯局部一步。

建议：

- `0.10 ~ 0.20`
- 当前推荐起点：`0.15`

### `--episode-settlement-decay`

- 作用：episode 结算往前分配时的衰减。
- 影响模型特征：决定越靠前的动作还能分到多少终局信用。

建议：

- `0.99 ~ 0.997`
- 当前推荐起点：`0.995`

### `--settlement-signal-mode`

- 作用：如何把整局 raw return 变成 episode 结算信号。
- 影响模型特征：
  - `mean` 更稳
  - `sum` 更激进
  - `sqrt` 介于两者之间

建议：

- 主训练用 `mean`

### `--settlement-normalization`

- 作用：历史分配时是否归一化。
- 影响模型特征：控制 retroactive bonus 总量是否固定。

建议：

- 主训练用 `sum`

### `--settlement-clip`

- 作用：限制单次 episode settlement 的绝对值上限。
- 影响模型特征：避免终局一次性把某局样本权重推爆。

建议：

- `4 ~ 8`
- 当前推荐起点：`6`

### `--map-route-defeat-gold-threshold`

- 作用：失败时剩余金币超过多少才追加路线惩罚。
- 影响模型特征：抑制“攒一堆钱不花就死”的路线。

建议：

- `140 ~ 180`

### `--map-route-defeat-gold-penalty-multiplier`

- 作用：剩余金币失败惩罚强度。
- 影响模型特征：模型是否学会把钱转成战力，而不是留着死。

建议：

- `1.5 ~ 3.0`
- 当前推荐起点：`2.0`

## 7. replay 采样偏置参数

### `--turn-end-sample-bonus`
### `--combat-end-sample-bonus`
### `--act-end-sample-bonus`
### `--run-end-sample-bonus`

- 作用：提高某些边界样本被 replay 抽中的概率。
- 影响模型特征：强化模型对“阶段边界”样本的学习。

建议：

- 一般保持默认
- 如果你发现模型整局胜负学得慢，可以略微提高 `run-end-sample-bonus`

## 8. 行为合法性与兼容参数

这组参数不直接改变 reward，而是改变 agent 是否容易卡 UI、是否会提前做不合适的动作。

### `--card-select-candidate-limit`

- 作用：升级/附魔等 overlay 的候选裁剪 top-k。
- 影响模型特征：控制选牌分支数。

建议：

- `6` 是合适起点
- 太大容易在卡牌界面浪费时间

### `--allow-premature-end-turn`

- 作用：允许在仍有可打牌时提前暴露 `end turn`。
- 影响模型特征：更自由，但更容易乱空过。

建议：

- 正常训练不要开

### `--end-turn-guard-stall-limit`
### `--end-turn-guard-timeout-seconds`

- 作用：结束回合保护机制。
- 影响模型特征：减少“还没认真想就空过”的问题。

建议：

- `end-turn-guard-timeout-seconds`: `2.5 ~ 3.0`

### `--allow-unseen-card-reward-proceed`

- 作用：在奖励窗口还没认真看卡前就允许直接 proceed。
- 影响模型特征：会损失卡牌奖励数据质量。

建议：

- 正常训练不要开

### `--card-reward-preview-guard-timeout-seconds`

- 作用：卡牌奖励预览保护时间。
- 影响模型特征：减少奖励界面草率跳过。

建议：

- `2.5 ~ 3.0`

## 9. 日志与运行参数

### `--enable-episode-logs`

- 作用：开启 episode 日志存档。
- 影响模型特征：不直接影响训练，但非常重要，便于离线复盘和再训练。

### `--episode-log-dir`

- 作用：日志目录。

### `--episode-log-model-name`

- 作用：文件名中的模型名标签。
- 适合用来区分不同实验。

### `--character`
### `--ascension`

- 作用：自动开局角色和难度。

### `--manual-start`

- 作用：关闭自动开局，改为手动。

### `--seed`

- 作用：随机种子。
- 用于复现实验。

## 10. 当前版本里“固定但重要”的非参数机制

这些行为不是主参数，但你训练时必须知道：

1. 低于 70% HP 的篝火会被硬性约束优先回血
2. checkpoint 会保存 `episode_index`
3. `.logs` 文件名格式为：

```text
<角色>-<最高层数>-<ep轮次>-<日期>-<时间>-<模型>.logs
```

## 11. 不要再作为主配置使用的旧参数

以下参数当前只是兼容旧命令，不建议再拿来写主训练脚本：

- `--updates-per-step`
- `--turn-settlement-weight`
- `--turn-settlement-clean-cap`
- `--turn-settlement-skip-penalty`
- `--combat-settlement-weight`
- `--act-settlement-weight`
- `--run-settlement-weight`
- `--turn-settlement-decay`
- `--combat-settlement-decay`
- `--act-settlement-decay`
- `--run-settlement-decay`

原因不是它们报错，而是现在的主训练语义已经变成 **episode-end settlement + updates-per-episode**。

## 12. 调参顺序建议

推荐按下面顺序调，不要一口气全改：

1. 先固定合法性与 UI 保护参数
   - `card-select-candidate-limit`
   - `end-turn-guard-timeout-seconds`
   - `card-reward-preview-guard-timeout-seconds`
2. 再固定 reward 主骨架
   - `floor-advance-weight`
   - `hp-delta-weight`
   - `combat-end-weight`
   - `run-defeat-weight`
   - `turn-skip-unspent-penalty`
   - `episode-settlement-weight`
3. 然后调搜索与探索
   - `simulations`
   - `combat-selection-simulations`
   - `temperature`
4. 最后再调学习强度
   - `learning-rate`
   - `updates-per-episode`
   - `replay-capacity`
   - `warmup-samples`

## 13. 症状到参数的快速映射

### 症状：模型很莽，只顾冲层，血线管理差

优先看：

- 降低 `floor-advance-weight`
- 提高 `hp-delta-weight`
- 提高 `combat-defeat-weight` 的绝对值
- 降低 `temperature`

### 症状：模型老是空过回合

优先看：

- 显式设置 `--turn-skip-unspent-penalty`
- 不要依赖 multiplier
- 保持 `--allow-premature-end-turn` 关闭

### 症状：模型太保守，打得慢，输出不足

优先看：

- 降低 `hp-delta-weight`
- 略提高 `enemy-hp-delta-weight`
- 略提高 `combat-tactical-shaping`

### 症状：训练波动大，loss 时高时低

优先看：

- 降低 `learning-rate`
- 降低 `updates-per-episode`
- 提高 `warmup-samples`
- 提高 `replay-capacity`
- 降低 `settlement-clip`

### 症状：模型学得太慢

优先看：

- 增加 `updates-per-episode`
- 略降 `warmup-samples`
- 适度提高 `learning-rate`

## 14. 参数示例

下面给 4 套。

### 14.1 稳定主训练版

适合当前版本长期跑。

```powershell
python .\run_training.py `
--max-steps 50000 `
--simulations 40 `
--combat-selection-simulations 10 `
--combat-selection-search-time-budget-seconds 0.03 `
--learning-rate 1.5e-4 `
--discount 0.997 `
--updates-per-episode 8 `
--replay-capacity 20000 `
--warmup-samples 1500 `
--temperature 0.45 `
--checkpoint-path checkpoints\ep_main_v1.json `
--checkpoint-interval 250 `
--character Ironclad `
--ascension 0 `
--floor-advance-weight 0.10 `
--act-advance-weight 6.0 `
--hp-delta-weight 5.0 `
--gold-delta-weight 0.25 `
--gold-delta-scale 25 `
--enemy-hp-delta-weight 8.0 `
--turn-end-weight 0.0 `
--turn-skip-unspent-penalty 6.0 `
--combat-end-weight 6.0 `
--combat-defeat-weight -10.0 `
--act-end-weight 2.0 `
--run-victory-weight 12.0 `
--run-defeat-weight -18.0 `
--combat-tactical-shaping 0.30 `
--end-turn-slack-penalty 0.0 `
--episode-settlement-weight 0.15 `
--episode-settlement-decay 0.995 `
--settlement-signal-mode mean `
--settlement-normalization sum `
--settlement-clip 6 `
--card-select-candidate-limit 6 `
--end-turn-guard-timeout-seconds 2.5 `
--card-reward-preview-guard-timeout-seconds 2.5 `
--map-route-defeat-gold-threshold 150 `
--map-route-defeat-gold-penalty-multiplier 2.0 `
--enable-episode-logs `
--resume
```

### 14.2 高探索数据采集版

适合先多收集 logs、让策略更发散。

```powershell
python .\run_training.py `
--max-steps 50000 `
--simulations 28 `
--combat-selection-simulations 8 `
--combat-selection-search-time-budget-seconds 0.02 `
--learning-rate 1.2e-4 `
--discount 0.997 `
--updates-per-episode 4 `
--replay-capacity 24000 `
--warmup-samples 2000 `
--temperature 0.65 `
--checkpoint-path checkpoints\ep_explore_v1.json `
--checkpoint-interval 300 `
--character Ironclad `
--ascension 0 `
--floor-advance-weight 0.08 `
--act-advance-weight 5.0 `
--hp-delta-weight 4.5 `
--gold-delta-weight 0.30 `
--gold-delta-scale 25 `
--enemy-hp-delta-weight 7.5 `
--turn-end-weight 0.0 `
--turn-skip-unspent-penalty 5.0 `
--combat-end-weight 5.0 `
--combat-defeat-weight -10.0 `
--act-end-weight 1.5 `
--run-victory-weight 10.0 `
--run-defeat-weight -16.0 `
--combat-tactical-shaping 0.28 `
--end-turn-slack-penalty 0.0 `
--episode-settlement-weight 0.12 `
--episode-settlement-decay 0.995 `
--settlement-signal-mode mean `
--settlement-normalization sum `
--settlement-clip 5 `
--card-select-candidate-limit 6 `
--end-turn-guard-timeout-seconds 2.5 `
--card-reward-preview-guard-timeout-seconds 2.5 `
--map-route-defeat-gold-threshold 160 `
--map-route-defeat-gold-penalty-multiplier 1.5 `
--enable-episode-logs `
--resume
```

### 14.3 快速冒烟 / 调 bug 版

适合验证界面逻辑、日志是否正确，不追求训练质量。

```powershell
python .\run_training.py `
--max-steps 5000 `
--simulations 16 `
--combat-selection-simulations 6 `
--combat-selection-search-time-budget-seconds 0.02 `
--learning-rate 2e-4 `
--discount 0.997 `
--updates-per-episode 2 `
--replay-capacity 4000 `
--warmup-samples 200 `
--temperature 0.55 `
--checkpoint-path checkpoints\ep_smoke_v1.json `
--checkpoint-interval 50 `
--character Ironclad `
--ascension 0 `
--floor-advance-weight 0.08 `
--act-advance-weight 5.0 `
--hp-delta-weight 5.0 `
--gold-delta-weight 0.25 `
--gold-delta-scale 25 `
--enemy-hp-delta-weight 7.0 `
--turn-end-weight 0.0 `
--turn-skip-unspent-penalty 5.0 `
--combat-end-weight 5.0 `
--combat-defeat-weight -10.0 `
--act-end-weight 1.0 `
--run-victory-weight 10.0 `
--run-defeat-weight -15.0 `
--combat-tactical-shaping 0.25 `
--end-turn-slack-penalty 0.0 `
--episode-settlement-weight 0.10 `
--episode-settlement-decay 0.995 `
--settlement-signal-mode mean `
--settlement-normalization sum `
--settlement-clip 4 `
--card-select-candidate-limit 6 `
--end-turn-guard-timeout-seconds 2.0 `
--card-reward-preview-guard-timeout-seconds 2.0 `
--map-route-defeat-gold-threshold 150 `
--map-route-defeat-gold-penalty-multiplier 1.5 `
--enable-episode-logs `
--resume
```

### 14.4 收敛 / 复训版

适合在已经有一定 checkpoint 基础上继续压缩随机性。

```powershell
python .\run_training.py `
--max-steps 50000 `
--simulations 56 `
--combat-selection-simulations 12 `
--combat-selection-search-time-budget-seconds 0.05 `
--learning-rate 1e-4 `
--discount 0.997 `
--updates-per-episode 12 `
--replay-capacity 30000 `
--warmup-samples 2500 `
--temperature 0.20 `
--checkpoint-path checkpoints\ep_converge_v1.json `
--checkpoint-interval 300 `
--character Ironclad `
--ascension 0 `
--floor-advance-weight 0.10 `
--act-advance-weight 6.0 `
--hp-delta-weight 5.5 `
--gold-delta-weight 0.20 `
--gold-delta-scale 25 `
--enemy-hp-delta-weight 8.0 `
--turn-end-weight 0.0 `
--turn-skip-unspent-penalty 6.0 `
--combat-end-weight 6.0 `
--combat-defeat-weight -11.0 `
--act-end-weight 2.0 `
--run-victory-weight 12.0 `
--run-defeat-weight -18.0 `
--combat-tactical-shaping 0.30 `
--end-turn-slack-penalty 0.0 `
--episode-settlement-weight 0.18 `
--episode-settlement-decay 0.996 `
--settlement-signal-mode mean `
--settlement-normalization sum `
--settlement-clip 6 `
--card-select-candidate-limit 6 `
--end-turn-guard-timeout-seconds 3.0 `
--card-reward-preview-guard-timeout-seconds 3.0 `
--map-route-defeat-gold-threshold 150 `
--map-route-defeat-gold-penalty-multiplier 2.0 `
--enable-episode-logs `
--resume `
--no-root-noise
```

## 15. 推荐起步方式

如果你现在准备重新开训练，建议这样用：

1. 先用“稳定主训练版”跑
2. 观察 20 ~ 50 个 episode 的 logs
3. 如果行为太发散，再降 `temperature`
4. 如果学习太慢，再加 `updates-per-episode`
5. 如果 reward 抖动太大，先降 `learning-rate` 和 `settlement-clip`

不要一开始就同时改 8 个以上参数，不然你无法判断到底是哪一个在起作用。
