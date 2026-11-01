# v0.1 — 任务证据与干预机制人工审查包

这是已审计 `TEN_CASE_PAIRED_AUDITED` 与已审计离线收益标签之后的**只读开发集分析**。仍在原有分支 `exp/intervention-dataset-native-gates-v01`，不更改任何原始仿真结果、采样方案、模型或 Manifest；无需 GPU。

## 审查对象

以下四个 Episode 是**事后根据已观察的收益与损害选出的定性分析对象**，不是新的盲选训练集，不得将选择结果当作在线模型输入，也不能报告为独立的泛化评估：

| Episode | 为什么审查 |
|---|---|
| i5noydFURQK/70 | 扰动后 KEEP 失败，RIGHT45 SEEK 成功 |
| r1Q1Z4BcV1o/106 | 扰动后 KEEP 失败，RIGHT45 SEEK 成功；自然状态下 RESET 有损害 |
| rPc6DW4iMge/73 | 自然和扰动状态下 RESET 均损害 |
| PX4nDJXEHrG/79 | 自然和扰动状态下 RESET 均损害，扰动状态 LEFT45 也损害 |

每个 Episode 都保留自然/扰动两个状态及 KEEP、SEEK_LEFT45、SEEK_RIGHT45、HISTORY_RESET 四臂，总共 **4 场景 / 8 状态 / 32 条已完成真实续航**。

## 在服务器执行（CPU）

~~~bash
cd ~/InternNav
git switch exp/intervention-dataset-native-gates-v01
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_intervention_selection_mechanism_review_v01.py' -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_mechanism_review_v01.py build \
  --label-dir data/intervention_v01_offline_labels_01 \
  --out-dir data/intervention_v01_mechanism_review_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_mechanism_review_v01.py audit \
  --review-dir data/intervention_v01_mechanism_review_01
~~~

输出文件夹必须不存在。若成功，状态为 `POSTHOC_MECHANISM_EVIDENCE_REVIEW_READY_NOT_ANNOTATED`，审计输出应为 `POSTHOC_EVIDENCE_REVIEW_AUDITED 4 scenes 8 contexts 32 real arms`。这里是目标输出，不代表已在你的服务器上运行。

## 输出与证据约束

- `review_contexts.jsonl`：8 条真实指令、主循环 step、动作前缀、当前缩放 RGB 与采样历史 RGB 的**原始绝对路径和 SHA256**，不包括干预后成功率/距离或位姿。只提供在线合法证据。
- `review_arm_evidence.jsonl`：32 条离线实验结果、真实动作轨迹的源文件路径、gate 之后的 STOP action-code 0 的位置、SCAN 真实外望 RGB、严格保存的首次 System 2 文本，以及 RESET 后历史索引过滤事件。带有 `OFFLINE_ONLY` 后缀的数据（包括真实位姿）**不可输入到选择模型**。
- `human_annotation_templates.jsonl`：8 个待人工判断的结构化模板。进度、地标、一致性、STOP 含义与候选机制默认 `UNDETERMINED` / `NOT_IDENTIFIED`，不是自动推断或训练真值。
- `review_report.json`：四个场景的显式事后选择依据、源数据/证据图像的 SHA256、输出哈希和样本数。

**注意限制：** 扰动 KEEP 的首次 S2 生成文本没有被原始 probe 单独保存，输出明确标注 `NOT_CAPTURED_IN_PERTURBED_KEEP`。原生 TRAIN 录制器中若存在自然 gate 的对应普通 S2 调用，只以 `ORIGINAL_NATIVE_SOURCE_DECISION_AT_NATURAL_GATE` 引用，不能冒充扰动 KEEP 的输出。

真实 action code 0 只说明 Habitat 收到了 STOP action。原始 Evaluator 某些分支也用 `action = 0` 作为默认值，不能直接将其解释为模型理解了终点。即使某个 Episode 的 KEEP 有 `oracle_success=1`，也不能由此断定 SEEK 改变的就是停止时机。

## 下一步人工判读程序

1. 对每个 context 阅读导航指令，查看当前 RGB 与按时间排序的真实历史 RGB，标出可见地标和可能完成的指令片段。无法从 RGB 确定的部分必须标记不确定。
2. 核对 SEEK 的外望真实 RGB、注入位置、首次 S2 输出和其后动作；比较 LEFT/RIGHT 方向及同状态 RESET/KEEP 的续航。
3. 将环境 STOP 动作位置与结果并列查看。区分：新观察提供有效信息、原始物理旋转改变后续控制、模型对目标完成进度判断变化等**待验证的竞争解释**。
4. RESET 仅修改被采样的历史 RGB 索引，并非清除模型内部全部记忆；判断损害可能来自哪些原本有用的历史地标，而不是直接称作“历史漂移已证实”。
5. 记录证据与反例。只有通过额外的 PLACEBO、SCAN_NO_INJECTION 等控制后，才能尝试做更明确的因果归因；本数据尚不足以证明物理扫描的信息是恢复原因。

本阶段不更新 `contexts_predecision.jsonl`、原始 `intervention_outcomes.jsonl`、离线标签或任何已冻结的仿真目录。人工填报请另存一个新版本；直接改动上面带原始哈希的证据包会导致审计失败。
