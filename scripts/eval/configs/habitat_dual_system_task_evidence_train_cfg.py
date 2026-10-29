"""Same frozen DualVLN configuration as habitat_dual_system_cfg; train data only."""

import copy

from scripts.eval.configs.habitat_dual_system_cfg import eval_cfg as original

eval_cfg = copy.deepcopy(original)
eval_cfg.env.env_settings["config_path"] = (
    "scripts/eval/configs/vln_r2r_task_evidence_train.yaml"
)
eval_cfg.agent.model_settings["kv_cache_continuation"] = False
eval_cfg.agent.model_settings["vis_debug"] = False
