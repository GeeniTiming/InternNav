"""Task-evidence v1: complete *native* DualVLN System-2 decision capture.

Prepare selects the first episode per distinct R2R TRAIN scene, without
inspecting outcomes. Run calls the original model/evaluator and logs every
System-2 invocation (including camera LOOKDOWN follow-ups) using guarded,
process-local AST hooks. The policy, navigation, history and prompt are NOT
modified. Native actions/poses are also written by cache_pilot's control arm.

Only public-to-agent visual/linguistic history enters capture/online.
Simulator poses and final metrics are OFFLINE_ONLY in separate files.
No evidence labels are inferred and no intervention/recovery is performed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_pilot as pilot
import failure_case_probe as capture

SCHEMA = "task_evidence_decisions.v1"
SELECTOR_FIELDS = (
    "episode_key", "instruction", "main_step", "call_type",
    "native_action_prefix_count", "history_native_indices",
    "images", "current_image_ref", "remaining_main_step_budget",
)
EXCLUDED_SELECTOR_FIELDS = (
    "sim_position", "sim_orientation", "distance_to_goal", "success",
    "reference_path", "future_actions", "oracle_success", "depth", "gps",
)
CONFIG = "scripts/eval/configs/habitat_dual_system_task_evidence_train_cfg.py"


def select_train_scenes(dataset, count):
    """Select one first route per scene in annotation order, outcome-blind."""
    if not 1 <= int(count) <= 100:
        raise ValueError("Pilot requires 1 to 100 independently selected scenes")
    scenes, selected, keys = set(), [], set()
    for ep in dataset["episodes"]:
        key = capture.episode_key(ep)
        ins = ep.get("instruction", {}).get("instruction_text", "").strip()
        if not ins or key[0] in scenes:
            continue
        if key in keys:
            raise ValueError("Duplicate episode key in source data")
        scenes.add(key[0])
        keys.add(key)
        selected.append(copy.deepcopy(ep))
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Need {count} distinct train scenes; only {len(selected)} are present"
        )
    subset = copy.deepcopy(dataset)
    subset["episodes"] = selected
    return subset


def require_train(cfg):
    import yaml
    if (cfg.eval_type != "habitat_vln"
            or cfg.agent.model_settings.get("mode") != "dual_system"
            or cfg.agent.model_settings.get("kv_cache_continuation", False)):
        raise ValueError("Require frozen, cache-disabled native DualVLN")
    path = (pilot.ROOT / cfg.env.env_settings["config_path"]).resolve()
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    split = obj["habitat"]["dataset"]["split"]
    if split != "train":
        raise ValueError("Evidence acquisition must use train split; got " + str(split))
    return path


def prepare(args):
    root = Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError("Create-only acquisition run: " + str(root))
    if args.episodes < 1 or args.episodes > 100:
        raise ValueError("Pilot episode count must be 1..100")
    cfg = pilot.load_config(pilot.ROOT / CONFIG)
    require_train(cfg)
    original = pilot.select_episodes
    try:
        pilot.select_episodes = select_train_scenes
        pilot.prepare(SimpleNamespace(
            run_dir=str(root), config=CONFIG, episodes=args.episodes,
            seed=args.seed, max_steps=args.max_steps,
        ))
    finally:
        pilot.select_episodes = original

    manifest = pilot.read_json(root / "manifest.json")
    manifest["selection"] = (
        "First instruction-bearing episode per distinct TRAIN scene in source "
        "annotation order; outcome-blind, no goal/reference path filtering."
    )
    manifest["task_evidence_capture"] = {
        "schema": SCHEMA, "split": "train", "episodes": int(args.episodes),
        "one_episode_per_scene": True, "capture": "every actual native System-2 generate",
        "no_intervention": True, "annotation_status": "not_annotated",
        "main_loop_limit": int(pilot.load_config(root / "control.py").eval_settings["max_steps_per_episode"]),
        "selector_fields": list(SELECTOR_FIELDS),
        "forbidden_selector_fields": list(EXCLUDED_SELECTOR_FIELDS),
    }
    manifest["source_hashes"][str(Path(__file__).resolve())] = pilot.sha256(__file__)
    pilot.write_json(root / "manifest.json", manifest)
    pilot.verify_manifest(root)
    print("TASK_EVIDENCE_PREPARE=PASS", args.episodes,
          "train scenes; no model loaded; output:", root)


def pixel_digest(image):
    """Hash actual PIL pixel bytes, independent of PNG compression."""
    key = image.mode.encode("ascii") + b"\0" + str(image.size).encode("ascii")
    return hashlib.sha256(key + b"\0" + image.tobytes()).hexdigest()


def validate_online_row(row):
    if row["schema"] != SCHEMA:
        raise ValueError("Online row schema drift")
    if any(name in row for name in EXCLUDED_SELECTOR_FIELDS):
        raise ValueError("Privileged fields leaked to online decision row")
    if not row.get("images"):
        raise ValueError("Empty System-2 image input")
    if row["call_type"] not in ("ordinary", "lookdown_followup"):
        raise ValueError("Unknown native System-2 call type")
    if row["call_type"] == "ordinary":
        idx = row["history_native_indices"]
        if len(row["images"]) != len(idx) + 1 or idx != sorted(set(idx)):
            raise ValueError("Ordinary System-2 history/img mapping changed")
        if row["current_image_ref"] != row["images"][-1]["path"]:
            raise ValueError("Current RGB is not last native input image")
    elif row["current_image_ref"] is not None:
        raise ValueError("LOOKDOWN follow-up is not an ordinary current RGB")
    if int(row["native_action_prefix_count"]) < 0:
        raise ValueError("Negative action prefix count")
    return True


def instrument_loop(function):
    """Inject two read-only event callbacks; fail closed on source drift."""
    if function.__code__.co_freevars:
        raise ValueError("Unexpected native evaluator closure")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    loops = [
        item for item in ast.walk(tree) if isinstance(item, ast.While)
        and any(isinstance(n, ast.Attribute) and n.attr == "max_steps_per_episode"
                for n in ast.walk(item.test))
    ]
    if len(loops) != 1:
        raise ValueError("Expected exactly one DualVLN main episode loop")
    loop = loops[0]
    expected_inputs = ast.parse(
        'inputs = self.processor(text=[text], images=input_images, '
        'return_tensors="pt").to(self.model.device)'
    ).body[0]
    expected_output = ast.parse(
        "llm_outputs = self.processor.tokenizer.decode("
        "output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)"
    ).body[0]
    n_in, n_out = 0, 0
    for item in ast.walk(loop):
        for _, block in ast.iter_fields(item):
            if not isinstance(block, list):
                continue
            for index, child in reversed(list(enumerate(block))):
                if isinstance(child, ast.Assign) and ast.dump(child) == ast.dump(expected_inputs):
                    block.insert(index + 1, ast.parse(
                        "self._evidence_recorder.before("
                        "step_id, action, episode_instruction, input_images, "
                        "history_id if action != action_code.LOOKDOWN else [], "
                        "len(rgb_list), sources[0]['value'], text)"
                    ).body[0])
                    n_in += 1
                elif isinstance(child, ast.Assign) and ast.dump(child) == ast.dump(expected_output):
                    block.insert(index + 1, ast.parse(
                        "self._evidence_recorder.after(step_id, llm_outputs)"
                    ).body[0])
                    n_out += 1
    if (n_in, n_out) != (1, 1):
        raise RuntimeError(
            f"NOT_COMPARABLE: native System-2 input/output points changed ({n_in}, {n_out})"
        )
    ast.fix_missing_locations(tree)
    scope = dict(function.__globals__)
    exec(compile(tree, "<native-task-evidence-logger>", "exec"), scope)
    return functools.update_wrapper(scope[function.__name__], function)


class DecisionRecorder:
    """Passive, write-once image catalog + one JSONL record per actual generate."""

    def __init__(self, evaluator, run_dir, manifest):
        self.evaluator, self.root, self.manifest = evaluator, Path(run_dir), manifest
        self.capture = self.root / "capture"
        self.capture.mkdir(exist_ok=False)
        self.images_dir = self.capture / "images"
        self.images_dir.mkdir(exist_ok=False)
        self.rows = (self.capture / "online_decisions.jsonl").open("x", encoding="utf-8")
        self.poses = (self.capture / "OFFLINE_ONLY_poses.jsonl").open("x", encoding="utf-8")
        self.original_step = evaluator.env.step
        self.action_count = 0
        self.next_call = 0
        self.pending = None
        self.by_episode = {}
        self.kinds = {"ordinary": 0, "lookdown_followup": 0}
        self.stored_images = set()
        self.status = "running"

    def step(self, action):
        result = self.original_step(action)
        self.action_count += 1
        return result

    def image_ref(self, image):
        if image.mode not in ("RGB", "L", "RGBA"):
            raise ValueError("Unexpected native System-2 image mode " + str(image.mode))
        digest = pixel_digest(image)
        path = self.images_dir / (digest + ".png")
        if digest not in self.stored_images:
            if path.exists():
                raise FileExistsError("Fresh image catalog unexpectedly already contains " + str(path))
            image.save(path, format="PNG")
            self.stored_images.add(digest)
        return {
            "path": "images/" + digest + ".png",
            "pixel_sha256": digest,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }

    def before(self, step, action, instruction, images, history_ids, rgb_count,
               prompt, rendered_prompt):
        if self.pending is not None:
            raise RuntimeError("System-2 call began before prior output was recorded")
        ep = self.evaluator.env.get_current_episode()
        key = capture.episode_key({"scene_id": ep.scene_id, "episode_id": ep.episode_id})
        if [ep.scene_id, str(ep.episode_id)] not in [
            [e["scene_id"], e["episode_id"]] for e in self.manifest["episodes"]
        ]:
            raise RuntimeError("Unexpected native episode in decision recorder " + str(key))
        call_type = ("lookdown_followup" if action == 5 else "ordinary")
        ids = [] if call_type == "lookdown_followup" else [int(v) for v in history_ids]
        if call_type == "ordinary":
            if any(v < 0 or v >= int(rgb_count) - 1 for v in ids):
                raise RuntimeError("History index outside observed RGB sequence")
            if len(images) != len(ids) + 1:
                raise RuntimeError("System-2 image count differs from native history")
        if str(instruction) != ep.instruction.instruction_text:
            raise RuntimeError("Native episode instruction changed")
        refs = [self.image_ref(image) for image in images]
        call_id = self.next_call
        self.next_call += 1
        budget = int(self.evaluator.max_steps_per_episode) - int(step)
        row = {
            "schema": SCHEMA,
            "call_id": call_id,
            "episode_key": list(key),
            "instruction": str(instruction),
            "main_step": int(step),
            "call_type": call_type,
            "native_action_prefix_count": self.action_count,
            "history_native_indices": ids,
            "images": refs,
            "current_image_ref": refs[-1]["path"] if call_type == "ordinary" else None,
            "remaining_main_step_budget": budget,
            "prompt_text": str(prompt),
            "rendered_prompt": str(rendered_prompt),
            "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
        }
        validate_online_row(row)
        # Pose is metadata for offline trajectory auditing ONLY. Never copied
        # into annotation templates or the online feature rows.
        state = self.evaluator.env._env.sim.get_agent_state()
        self.poses.write(json.dumps({
            "call_id": call_id, "episode_key": list(key),
            "sim_position": [float(v) for v in state.position],
            "sim_orientation": [float(state.rotation.w), float(state.rotation.x),
                                float(state.rotation.y), float(state.rotation.z)],
        }, ensure_ascii=False) + "\n")
        self.poses.flush()
        self.pending = row

    def after(self, step, output):
        if self.pending is None or int(step) != self.pending["main_step"]:
            raise RuntimeError("Unpaired System-2 response or main step changed")
        row = self.pending
        row["model_output"] = str(output)
        row["native_parsed_actions"] = [int(v) for v in self.evaluator.parse_actions(str(output))]
        validate_online_row(row)
        self.rows.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows.flush()
        key = tuple(row["episode_key"])
        self.by_episode[key] = self.by_episode.get(key, 0) + 1
        self.kinds[row["call_type"]] += 1
        self.pending = None

    def close(self, error=None):
        self.status = "failed" if error else "complete"
        if self.pending is not None:
            self.status = "failed"
            error = error or "Partial System-2 call has no decoded output"
        self.rows.close()
        self.poses.close()
        pilot.write_json(self.capture / "capture_summary.json", {
            "schema": SCHEMA,
            "status": self.status,
            "error": error,
            "N_decisions": self.next_call - int(self.pending is not None),
            "N_unique_images": len(self.stored_images),
            "N_episodes_with_decisions": len(self.by_episode),
            "N_actual_env_actions": self.action_count,
            "call_types": self.kinds,
            "episodes": [{
                "key": list(k), "decisions": n
            } for k, n in sorted(self.by_episode.items())],
            "safety": "Native-only; simulator pose in OFFLINE_ONLY file, never online",
        })


def run(args):
    root = Path(args.run_dir).resolve()
    manifest = pilot.verify_manifest(root)
    if manifest.get("task_evidence_capture", {}).get("schema") != SCHEMA:
        raise ValueError("Not a task-evidence acquisition preparation")
    cfg = pilot.load_config(root / "control.py")
    require_train(cfg)
    if (root / "capture").exists() or (root / "control").exists():
        raise FileExistsError("Create-only capture and control directories already exist")
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators["habitat_vln"]
    original_init, original_loop = cls.__init__, cls._run_eval_dual_system
    recorders = []

    def init(evaluator, config):
        original_init(evaluator, config)
        rec = DecisionRecorder(evaluator, root, manifest)
        recorders.append(rec)
        evaluator._evidence_recorder = rec
        evaluator.env.step = rec.step

    cls.__init__, cls._run_eval_dual_system = init, instrument_loop(original_loop)
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm="control", profile_stages=False))
        if len(recorders) != 1:
            raise RuntimeError("Expected one native evaluator, got " + str(len(recorders)))
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = original_init, original_loop
        for rec in recorders:
            rec.evaluator.env.step = rec.original_step
            rec.close(error)
    return report(SimpleNamespace(run_dir=str(root)))


def read_records(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(t) for t in f if t.strip()]


def report(args):
    root = Path(args.run_dir).resolve()
    manifest = pilot.verify_manifest(root)
    cp = manifest.get("task_evidence_capture", {})
    if cp.get("schema") != SCHEMA or cp.get("split") != "train":
        raise ValueError("Not a complete TRAIN decision acquisition")
    native = pilot.read_json(root / "control/pilot_summary.json")
    summary = pilot.read_json(root / "capture/capture_summary.json")
    if native["status"] != "complete" or summary["status"] != "complete":
        raise ValueError("Incomplete native run/capture")
    records = read_records(root / "capture/online_decisions.jsonl")
    poses = read_records(root / "capture/OFFLINE_ONLY_poses.jsonl")
    actions = read_records(root / "control/actions.jsonl")
    progress = capture.read_progress(root / "control/progress.json")
    expected = {capture.episode_key(x) for x in manifest["episodes"]}
    if set(progress) != expected or len(expected) != cp["episodes"]:
        raise ValueError("Native episode set changed or an episode is missing")
    if (len(actions) != summary["N_actual_env_actions"]
            or len(records) != summary["N_decisions"]
            or len(records) != len(poses)
            or len(actions) != int(native["actual_action_count"])):
        raise ValueError("Native traces and System-2 records are inconsistent")
    if set(tuple(x["key"]) for x in summary["episodes"]) != expected:
        raise ValueError("One or more selected scenes have no System-2 decisions")
    images = set()
    for i, (row, pose_row) in enumerate(zip(records, poses)):
        validate_online_row(row)
        if row["call_id"] != i or pose_row["call_id"] != i:
            raise ValueError("S2 call IDs are not contiguous")
        if tuple(row["episode_key"]) not in expected or row["episode_key"] != pose_row["episode_key"]:
            raise ValueError("Mismatched scene or offline-only position")
        if not row["model_output"]:
            raise ValueError("Empty System-2 model output")
        if row["remaining_main_step_budget"] < 0:
            raise ValueError("Negative System-2 execution budget")
        for ref in row["images"]:
            p = root / "capture" / ref["path"]
            if not p.is_file():
                raise FileNotFoundError(p)
            images.add(ref["path"])
    if len(images) != summary["N_unique_images"]:
        raise ValueError("Stored image count mismatch")
    # Publish only past action codes as an online-legal timeline, never poses.
    # Every decision references the index *before* its System-2 generation.
    action_only = root / "capture/online_action_codes.jsonl"
    with action_only.open("w", encoding="utf-8") as f:
        for index, row in enumerate(actions):
            f.write(json.dumps({
                "native_action_index": index,
                "episode_key": list(capture.episode_key(row)),
                "action": int(row["action"]),
            }, ensure_ascii=False) + "\n")
    for row in records:
        if row["native_action_prefix_count"] > len(actions):
            raise ValueError("System-2 decision references future native actions")
    results = {
        "schema": SCHEMA, "status": "TRAIN_NATIVE_DECISION_DATASET_READY",
        "split": "train", "N_episodes": len(expected),
        "N_decisions": len(records), "N_image_assets": len(images),
        "N_action_records": len(actions),
        "call_types": summary["call_types"],
        "selector_inputs_exclude_privileged_state": True,
        "label_status": "NOT_LABELED",
        "interventions_executed": False,
        "limitations": "No landmark or progress labels; no mechanism/causal evidence; no held-out evaluation.",
    }
    pilot.write_json(root / "capture/decision_report.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare", help="CPU-only locked train scene selection")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--max-steps", type=int, default=None)
    p.set_defaults(function=prepare)
    for name, func in (("run", run), ("report", report)):
        p = subs.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.set_defaults(function=func)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
