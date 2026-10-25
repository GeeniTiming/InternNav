# DualVLN KV-cache 小样本验证

这份分支集成了社区 [PR #368](https://github.com/InternRobotics/InternNav/pull/368)
的可选 KV-cache continuation，保留原作者提交历史；新增的只是独立实验入口和对照记录。
它解决 `generate_latents` 再次编码同一批图像和前缀的问题，**不是单卡多环境 batch，也不是新的导航恢复算法**。

社区只报告了两条受步数限制的 episode：整体时间从 39.63 秒降至 35.91 秒；
17.30 倍是局部 latent readout 的加速，不能解释为整套导航加速。
BF16 结果并非逐位相等，原 PR 尚未完成全量 SR/SPL 验证。
这些均为社区报告，本分支不将其当作我们自己完成的 GPU 验证。

## 现在可以做：在现有仓库切换分支、CPU 测试与实验准备

以下命令使用现有 `/home/geenitiming/InternNav`，不创建另一份项目目录。
继续使用同一个 conda 环境、数据和权重；实验结果写入独立的 run-dir。
切换分支会更新磁盘源码，已经启动的 Python 评测不会自动启用新缓存路径。
记录旧评测时仍应使用其启动时的代码版本，保留旧日志。

```bash
conda activate internnav_habitat
cd /home/geenitiming/InternNav
git switch perf/dualvln-cache-pilot
git pull --ff-only

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m pytest -q tests/unit_test/test_kv_cache_continuation.py tests/unit_test/test_cache_pilot.py

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' python scripts/eval/cache_pilot.py prepare \
  --run-dir data/cache_pilot_01 --episodes 1
```

实验目录使用创建操作；已有目录不会被强制覆盖。
重新准备另一轮实验时改用新的目录名，例如 `data/cache_pilot_02`。
准备步骤不初始化模型或仿真器，不运行导航。

`prepare` 从原始标注选取声明数量的前 N 条 episode，不读取结果、不按成功率筛选；
完整保留 `instruction_vocab`。首轮建议一条，验证通过后再准备更大的固定集合。
默认继承原评测配置，包括历史帧数、图像尺寸、扩散参数和 500 步上限。
`--max-steps` 仅用于显式声明的调试截断；截断结果不得充当完整基准结果。

## GPU 对照：两组顺序执行

当前原评测和另一用户的 `train` 在共享 GPU。可以先完成上面的 CPU 验证，
等合适的 GPU 时段再跑以下实验。共享卡上的结果会记录其他进程快照，
其耗时不代表独占卡性能。

```bash
cd /home/geenitiming/InternNav
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python -u scripts/eval/cache_pilot.py run \
  --run-dir data/cache_pilot_01 --arm control

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python -u scripts/eval/cache_pilot.py run \
  --run-dir data/cache_pilot_01 --arm cache

python scripts/eval/cache_pilot.py report --run-dir data/cache_pilot_01
```

每条 episode 在 reset 前设置同一个 Habitat/Python/NumPy/Torch 种子。
两组分别产生 `control/`、`cache/`，不会续跑或混入旧评测的 `progress.json`。
记录包括所有实际 Habitat 动作（含抬头/低头）、动作后位置和朝向、导航指标、
初始化以外的总时间、Torch 峰值显存及起止 GPU 进程快照。

主要结果是 `comparison.json` 中的 `actions_and_poses_match`、完整指标和
`observed_time_ratio`。相同动作和位姿是小样本检查，不是全数据集等价证明；
指标相同也不能替代动作记录对照。位置/朝向默认容差为 `1e-5`，朝向比较处理四元数正负等价。
脚本会核查标注、配置、关键代码及模型配置/index 的 SHA256 和分片大小；
不会重复散列全部约 16 GB 的权重，因此这是同一路径权重的检查，不是完整权重内容认证。

如果动作出现差异，先检查并保留差异记录，不把改动直接用于正式全量结果。
即使一次完全一致，也仍需要更大的固定集合和完整 SR/SPL 验证。

## 定位瓶颈

在两组 `run` 命令中都加 `--profile-stages`，可以分别测 S2 生成、latent readout、
S1 轨迹生成与 Habitat step。该选项会在阶段边界同步 CUDA，存在额外计时开销；
它用于定位瓶颈。普通端到端对照默认关闭阶段同步，结果也包含少量动作日志开销。
两组 profile 设置不一致时，`report` 会拒绝比较。

性能比较应采用相同硬件负载与重复实验；起止进程快照无法证明整个运行期间独占 GPU。
本脚本不停止、限速或修改任何其他 GPU 进程。

## A/A 复现后，定位 A/B 动作差异

服务器本次单条 episode 的用户回传结果：关闭缓存的两次运行，308 个实际动作及位姿
完全一致；关闭/开启缓存的指标相同，但零基动作索引 71 首次发生前进/右转差异。
A/B 耗时为 43.329 / 37.944 秒；这是共享卡单次观测，不是稳定加速比例。
因此不能将缓存标记为行为等价，也不能仅凭这两个运行确定具体数值原因。

下面的旁路诊断只让原始 full-replay 策略执行动作。对每次相同 generation 的两种
latent 读出进行比较，再以相同 S1 随机状态分别生成轨迹；额外重复原始读出与原始 S1。
每次旁路调用后恢复 Python/NumPy/Torch/CUDA RNG，检查缓存长度、前缀内容及原张量版本；
解码动作时克隆轨迹，避免原解码器的原地缩放影响实际输出。
原有模型、评测器和 pilot 脚本不修改，旧实验文件不覆盖。

```bash
cd /home/geenitiming/InternNav
git pull --ff-only

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m pytest -q tests/unit_test/test_kv_cache_continuation.py \
  tests/unit_test/test_cache_pilot.py tests/unit_test/test_cache_readout_diagnostic.py

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/cache_pilot.py prepare --run-dir data/cache_pilot_03 --episodes 1

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/cache_readout_diagnostic.py \
  --run-dir data/cache_pilot_03 --reference-control data/cache_pilot_01/control
```

输出为 `data/cache_pilot_03/paired_diagnostic/summary.json` 及 `readouts.jsonl`、
`plans.jsonl`。`next_actual_action_index` 用于关联局部计划与实际动作序列，采用零基索引。
摘要包含原始执行轨迹是否仍匹配旧 control、原始读出/S1 重复是否逐位一致，以及
缓存表征是否改变了局部动作。`first_cache_plan_difference` 保存第一次前四动作差异。
追加和裁剪缓存允许更换底层存储地址；前缀检查比较全部保留的 key/value，
同时检查旧张量版本以发现原地修改。还须检查每次输出的有限性和误差。
这里每次 S1 只使用同一组真实图像、深度和参数，更换的是两种读出得到的 latent。

`paired_action_effect_evidence_valid=true` 要求原始轨迹匹配、原始读出和 S1 的重复计算
逐位一致且确实发生了配对计算；它不表示缓存没有问题。
即使该字段为 true，也只支持这个案例中固定输入的影响判断，不证明全量 SR/SPL 等价。
如果旁路不改变即时动作，还要检查跨后续时刻的累积差异，不能提前排除缓存问题。
此运行包含大量额外计算，`timing_valid_for_speed_comparison=false`；
不要把它的耗时与普通 control/cache 进行速度比较。

## 验证范围

本分支已在 Python 3.12 / PyTorch 2.6.0 CPU / Transformers 4.51.0 下通过
上述两个文件的 **34 项测试**；这不是用户服务器的 Python 3.9 / CUDA / BF16 验证。
新增旁路诊断另有 **7 项 CPU 测试**，覆盖原始输出与 RNG 的隔离、异常恢复、
解码器原地修改防护和真实 DynamicCache 追加/裁剪；三份测试共 41 项。

- 社区 tiny-model CPU 回归：默认路径、缓存长度、mRoPE、EOS、重复读取、异常清理及数值对照。
- 本分支 CPU 测试：实验选择、配置保持、输入变更检测、结果隔离、相机动作、位姿和四元数对照。
- 单卡大模型 BF16、FlashAttention、Habitat 完整闭环和实际速度：需要在服务器运行上述 GPU 对照。
- 全量导航质量、恢复算法贡献：本次工程修改不作结论。
