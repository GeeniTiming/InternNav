# 单条原生失败轨迹的采集与复现检查

当前只研究失败在哪里，不设计恢复算法。已有20条对照中有6条原生失败；
episode 4/5/6都涉及楼梯，episode 15的文字包含连续的室内地标，先选15做诊断。
选择依据是便于检查执行过程，不是恢复收益，也不声称它代表全部失败。
是否确实为平层路线、是否有语义歧义，还需要结合真实观测核实。

本入口复用已有 cache_pilot 和原生 vis_debug。不开缓存、不改动作、
不注入物理恢复、不训练。保留源配置、每episode种子及评测上限，
只开启调试视频并改用新的视频/结果目录。原有20条结果不会覆盖。
源指令必须精确匹配，源control必须已经完成且该episode确实失败。
源标注、配置、动作日志和元数据有指纹核查。

## 先在原仓库更新并做CPU检查

~~~bash
cd /home/geenitiming/InternNav
git pull --ff-only
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/unit_test -p test_failure_case_probe.py -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/failure_case_probe.py prepare \
  --source-run data/cache_pilot_05 \
  --run-dir data/failure_case15_01 \
  --scene zsNo4HB9uLZ --episode 15
~~~

prepare首先检查imageio的FFmpeg编码器可执行，再复用原有实验准备逻辑。
预期VIDEO_ENCODER_CHECK=PASS及FAILURE_PROBE_PREPARE=PASS。
如果编码器缺失，先解决该错误，不要进入GPU运行。
新目录只允许创建一次；重试另一轮时换用新目录，保留失败记录。

## 只运行一条原始策略，保存视频与控制台输出

~~~bash
set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/cache_pilot.py run \
  --run-dir data/failure_case15_01 --arm control \
  2>&1 | tee -a data/failure_case15_01/console.log

python scripts/eval/failure_case_probe.py report \
  --run-dir data/failure_case15_01
~~~

vis_debug与普通save_video是不同入口：前者在原生代码中也保存失败视频。
视频采用原生的步数、动作与像素目标标注，路径在report的videos字段中。
console.log包含原生S2输出、像素目标和局部动作日志；actions.jsonl记录全部实际
动作（含相机动作）、位置和朝向。原来的全量评测继续运行。

报告按场景和episode筛选源control，比较全部动作和位姿，不把不同episode拼接对齐。
同时核查逐条SR/SPL/OS/NE和运行时版本、种子设置；需要非空视频文件。
READY_FOR_VISUAL_REVIEW表示复现轨迹与源记录一致且找到了视频文件，
不表示已经解释了失败，也不表示自动验证了视频内容或任务语义。
如果NOT_COMPARABLE_OR_MISSING_VIDEO，先查看复现差异或缺失项。

得到报告后需要查看视频及console.log，定位第一次可确认的错误：
S2选择了不合适的目标、S1未执行好目标，还是结束判断不合适。
仅凭最终NE、OS或STOP不能判定根因。还没有恢复干预和记忆因果结论。
当前val_unseen案例只作探索性诊断；后续方法开发需另建训练集小集合。

## 本地验证范围

7项Python 3.12 CPU测试通过，覆盖案例唯一选择、指令一致性、种子和配置保持、
错误后的函数恢复、源文件变更检测、逐episode复现门槛及视频文件存在检查。
测试不需要torch、Habitat或模型权重；不代替服务器上的GPU、视频编码和真实闭环验证。
