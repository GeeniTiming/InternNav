# DualVLN KV-cache 小样本验证

这份分支集成了社区 [PR #368](https://github.com/InternRobotics/InternNav/pull/368)
的可选 KV-cache continuation，保留原作者提交历史；新增的只是独立实验入口和对照记录。
它解决 `generate_latents` 再次编码同一批图像和前缀的问题，**不是单卡多环境 batch，也不是新的导航恢复算法**。

社区只报告了两条受步数限制的 episode：整体时间从 39.63 秒降至 35.91 秒；
17.30 倍是局部 latent readout 的加速，不能解释为整套导航加速。
BF16 结果并非逐位相等，原 PR 尚未完成全量 SR/SPL 验证。
这些均为社区报告，本分支不将其当作我们自己完成的 GPU 验证。

## 现在可以做：独立工作目录、CPU 测试与实验准备

以下命令保留 `/home/geenitiming/InternNav` 中正在运行的任务和源码。
新工作目录使用同一个 conda 环境，数据及权重通过软链接访问，结果写入独立的 run-dir。
Python 搜索路径明确指向新工作目录，避免 editable install 意外导入旧目录的代码。

```bash
conda activate internnav_habitat
git -C /home/geenitiming/InternNav fetch https://github.com/GeeniTiming/InternNav.git perf/dualvln-cache-pilot
git -C /home/geenitiming/InternNav worktree add --detach /home/geenitiming/InternNav-cache-pilot FETCH_HEAD

ln -sT /home/geenitiming/InternNav/data /home/geenitiming/InternNav-cache-pilot/data
ln -sT /home/geenitiming/InternNav/checkpoints /home/geenitiming/InternNav-cache-pilot/checkpoints
cd /home/geenitiming/InternNav-cache-pilot

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m pytest -q tests/unit_test/test_kv_cache_continuation.py tests/unit_test/test_cache_pilot.py

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' python scripts/eval/cache_pilot.py prepare \
  --run-dir data/cache_pilot_01 --episodes 1
```

`worktree add`、软链接和实验目录均使用创建操作；已有目录不会被强制覆盖。
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
cd /home/geenitiming/InternNav-cache-pilot
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

## 验证范围

本分支已在 Python 3.12 / PyTorch 2.6.0 CPU / Transformers 4.51.0 下通过
上述两个文件的 **34 项测试**；这不是用户服务器的 Python 3.9 / CUDA / BF16 验证。

- 社区 tiny-model CPU 回归：默认路径、缓存长度、mRoPE、EOS、重复读取、异常清理及数值对照。
- 本分支 CPU 测试：实验选择、配置保持、输入变更检测、结果隔离、相机动作、位姿和四元数对照。
- 单卡大模型 BF16、FlashAttention、Habitat 完整闭环和实际速度：需要在服务器运行上述 GPU 对照。
- 全量导航质量、恢复算法贡献：本次工程修改不作结论。
