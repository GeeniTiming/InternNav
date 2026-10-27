"""Build paired 2x2 recovery labels from four *completed, locked* InternNav runs.

CPU-only and stdlib-only; does not run Habitat, modify source runs, create a
training/test split, or pretend that RGB history is contained in action logs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil

ARMS = {
    "00": ("release", 0, 0),
    "10": ("right90", 1, 0),
    "01": ("history_reset", 0, 1),
    "11": ("right90_reset", 1, 1),
}
SCHEMAS = {
    "checkpoint": "crosscase_orientation_checkpoints.v1",
    "turn": "orientation_turn_report.v1",
    "candidates": "recovery_candidates_report.v1",
}
REQUIRED_FIELDS = ("success", "spl", "os", "ne", "steps")


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def lines(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path, items):
    with Path(path).open("x", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def key_of(row):
    name = str(row["scene_id"]).replace("\\", "/")
    return (PurePosixPath(name).stem, str(row["episode_id"]))


def key_of_pair(row):
    key = tuple(str(x) for x in row["key"])
    if len(key) != 2:
        raise ValueError("Invalid [scene, episode] key")
    return key


def indexed_progress(path, expected):
    result = {}
    for row in lines(path):
        key = key_of(row)
        if key in result:
            raise ValueError("Duplicate progress episode: " + repr(key))
        for field in REQUIRED_FIELDS:
            if field not in row:
                raise ValueError("Missing metric " + field + " in " + str(path))
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError("Non-finite metric " + field)
        if float(row["success"]) not in (0., 1.) or float(row["os"]) not in (0., 1.):
            raise ValueError("Expected binary SR/OS for " + repr(key))
        result[key] = row
    if set(result) != expected:
        raise ValueError("Episode set mismatch: " + str(path))
    return result


def index_rows(rows, label, expected):
    out = {}
    for item in rows:
        key = key_of_pair(item)
        if key in out:
            raise ValueError("Duplicate " + label + " outcome: " + repr(key))
        out[key] = item
    if set(out) != expected:
        raise ValueError("Outcome set mismatch: " + label)
    return out


def assert_equal(got, wanted, message):
    if got != wanted:
        raise ValueError("NOT_COMPARABLE: " + message)


def assert_close(got, wanted, message, tolerance=1e-5):
    if not math.isclose(float(got), float(wanted), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError("NOT_COMPARABLE: " + message)


def check_boundary(path, gate, label):
    row = load_json(path)
    for field in ("step", "prefix_action_count", "history_length",
                  "history_sha256", "rng_sha256", "parameters_sha256",
                  "prefix_trace_sha256", "observation_hashes", "state"):
        assert_equal(row.get(field), gate[field], label + " boundary " + field)
    return row


def verify_input_structure(checkpoint_root, turn_root, candidate_root):
    cp_report = load_json(checkpoint_root / "checkpoint_report.json")
    turn_report = load_json(turn_root / "turn_report.json")
    cand_report = load_json(candidate_root / "candidate_report.json")
    for name, report in (("checkpoint", cp_report), ("turn", turn_report),
                         ("candidates", cand_report)):
        assert_equal(report.get("schema"), SCHEMAS[name], name + " report schema")
    assert_equal(cp_report.get("status"), "CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION",
                 "checkpoint report incomplete")
    assert_equal(turn_report.get("status"), "READY_FOR_REVIEW", "turn report incomplete")
    assert_equal(cand_report.get("status"), "READY_FOR_INTERPRETATION",
                 "candidate report incomplete")
    cases = cp_report["cases"]
    keys = [key_of_pair(c) for c in cases]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Empty or duplicate locked checkpoint cohort")
    expected = set(keys)
    for c in cases:
        assert_equal(c.get("status"), "COMPARABLE_CHECKPOINT_CAPTURED",
                     "checkpoint not comparable " + repr(key_of_pair(c)))
        if not c.get("checkpoint"):
            raise ValueError("Missing locked checkpoint")
        if c.get("native_trace_comparison", {}).get("actions_and_poses_match") is not True:
            raise ValueError("Unreproduced native control")
    turn_protocol = load_json(turn_root / "turn_protocol.json")
    cand_protocol = load_json(candidate_root / "candidate_protocol.json")
    assert_equal(turn_protocol["cases"], cases, "turn checkpoint protocol differs")
    assert_equal(cand_protocol["cases"], cases, "candidate checkpoint protocol differs")
    assert_equal(turn_protocol["forced_turn_actions"], 6, "turn action declaration")
    assert_equal(cand_protocol["arms"],
                 ["history_reset", "left90_reset", "right90_reset", "backtrack_replan"],
                 "candidate arm declaration")
    assert_equal(int(cand_protocol["native_main_step_limit"]),
                 int(turn_protocol["native_main_step_limit"]), "native step budget")
    assert_equal(int(turn_protocol["charged_main_step_limit"]),
                 int(turn_protocol["native_main_step_limit"]) - 6, "turn cost rule")
    src_turn = load_json(turn_root / "manifest.json")["orientation_turn"]["source_run"]
    src_cand = load_json(candidate_root / "manifest.json")["recovery_candidates"]["source_run"]
    assert_equal(Path(src_turn).resolve(), checkpoint_root.resolve(),
                 "turn run points to a different checkpoint source")
    assert_equal(Path(src_cand).resolve(), checkpoint_root.resolve(),
                 "candidate run points to a different checkpoint source")
    assert_equal(Path(cand_protocol["release_control_source"]).resolve(),
                 (checkpoint_root / "control").resolve(), "release source differs")
    if not cand_report.get("source_control_reused"):
        raise ValueError("Candidate report did not reuse frozen RELEASE control")
    turn_data = index_rows(turn_report["outcomes"]["right90"], "right90 report", expected)
    cand_data = {
        arm: index_rows([r for r in cand_report["outcomes"] if r["arm"] == arm],
                        arm + " report", expected)
        for arm in ("history_reset", "right90_reset")
    }
    progress = {
        "00": indexed_progress(checkpoint_root / "control/progress.json", expected),
        "10": indexed_progress(turn_root / "right90/progress.json", expected),
        "01": indexed_progress(candidate_root / "history_reset/progress.json", expected),
        "11": indexed_progress(candidate_root / "right90_reset/progress.json", expected),
    }
    return cases, progress, turn_data, cand_data, turn_protocol, cand_protocol


def make_dataset(checkpoint_root, turn_root, candidate_root, out_root):
    """Fail before any output is created on invalid or mismatched source data."""
    checkpoint_root = Path(checkpoint_root).resolve()
    turn_root = Path(turn_root).resolve()
    candidate_root = Path(candidate_root).resolve()
    out_root = Path(out_root).resolve()
    if out_root.exists():
        raise FileExistsError("Dataset export is create-only: " + str(out_root))
    cases, progress, turn_data, cand_data, turn_protocol, cand_protocol = (
        verify_input_structure(checkpoint_root, turn_root, candidate_root)
    )
    cp_traces = {}
    for r in lines(checkpoint_root / "control/actions.jsonl"):
        cp_traces.setdefault(key_of(r), []).append(r)
    actual_keys = {key_of_pair(c) for c in cases}
    assert_equal(set(cp_traces), actual_keys, "baseline action trace episode set")
    checked = []
    contexts, labels, long_rows = [], [], []
    for case in cases:
        key = key_of_pair(case)
        gate = case["checkpoint"]
        scene, episode = key
        safe = scene + "_" + episode
        cp_image = checkpoint_root / "control/checkpoints" / safe / "rgb.png"
        if not cp_image.is_file():
            raise FileNotFoundError("Missing actual checkpoint RGB: " + str(cp_image))
        cp_gate = checkpoint_root / "control/checkpoints" / safe / "checkpoint.json"
        check_boundary(cp_gate, gate, "control/" + safe)
        for root, arm in ((turn_root, "right90"),
                          (candidate_root, "history_reset"),
                          (candidate_root, "right90_reset")):
            check_boundary(root / arm / "checkpoints" / safe / "boundary.json",
                           gate, arm + "/" + safe)
        prefix_count = int(gate["prefix_action_count"])
        trace = cp_traces[key]
        if prefix_count < 1 or prefix_count > len(trace):
            raise ValueError("Invalid prefix action count: " + repr(key))
        if not all("action" in r for r in trace[:prefix_count]):
            raise ValueError("Missing action in native trace")
        instruction = progress["00"][key]["episode_instruction"]
        metrics = {}
        for arm_id, (arm, physical, memory) in ARMS.items():
            result = progress[arm_id][key]
            assert_equal(result["episode_instruction"], instruction,
                         "instruction diverged " + repr((key, arm)))
            metrics[arm_id] = {
                "success": int(float(result["success"])),
                "os": int(float(result["os"])),
                "spl": float(result["spl"]),
                "ne_m": float(result["ne"]),
                "native_main_steps": int(result["steps"]),
            }
        r10, r01, r11 = turn_data[key], cand_data["history_reset"][key], cand_data["right90_reset"][key]
        for field, code, report in (("success", "10", r10),
                                    ("success", "01", r01), ("success", "11", r11)):
            assert_close(report[field], metrics[code][field], "report/progress SR mismatch")
        assert_close(r10["control_success"], metrics["00"]["success"], "turn/control SR")
        assert_close(r10["control_ne"], metrics["00"]["ne_m"], "turn/control NE")
        for code, detail in (("01", r01), ("11", r11)):
            assert_close(detail["baseline_success"], metrics["00"]["success"],
                         code + " baseline success")
            assert_close(detail["baseline_os"], metrics["00"]["os"], code + " baseline OS")
            assert_close(detail["baseline_ne"], metrics["00"]["ne_m"], code + " baseline NE")
            assert_close(detail["ne"], metrics[code]["ne_m"], code + " progress NE")
        assert_equal(int(r10["forced_turns"]), 6, "right90 turn count")
        assert_equal(int(r01["external_actions"]), 0, "reset-only must not move")
        assert_equal(int(r11["external_actions"]), 6, "joint physical action cost")
        for code, detail in (("01", r01), ("11", r11)):
            if int(detail["history_prompt_events"]) <= 0:
                raise ValueError("No post-handoff reset history prompts: " + repr(key))
            history_path = candidate_root / ARMS[code][0] / "checkpoints" / safe / "history_inputs.json"
            histories = load_json(history_path)
            assert_equal(len(histories), int(detail["history_prompt_events"]),
                         "history prompt event count mismatch")
            for h in histories:
                ids = h["selected_history_ids"]
                if any(int(i) < int(h["cutoff"]) or
                       int(i) >= int(h["rgb_count"]) - 1 for i in ids):
                    raise ValueError("Pre-handoff history leaked into reset prompt")
        outcomes = {arm: metrics[arm]["success"] for arm in ARMS}
        gamma = outcomes["11"] - outcomes["10"] - outcomes["01"] + outcomes["00"]
        assert gamma in (-2, -1, 0, 1, 2)
        costs = {"00": 0, "10": 6,
                 "01": int(r01["external_actions"]),
                 "11": int(r11["external_actions"])}
        contexts.append({
            "key": list(key), "context_id": safe, "split": "exploratory_development_only",
            "instruction": instruction, "checkpoint_main_step": int(gate["step"]),
            "native_main_step_limit": int(cand_protocol["native_main_step_limit"]),
            "remaining_main_loop_budget_proxy": int(cand_protocol["native_main_step_limit"]) - int(gate["step"]),
            "checkpoint_rgb_relative_path": "contexts/" + safe + "/current_rgb.png",
            "history_action_relative_path": "contexts/" + safe + "/prefix_actions.jsonl",
            "history_action_count": prefix_count,
            "history_rgb_available": False, "history_rgb_paths": [],
            "ready_for_full_multimodal_training": False,
            "history_missing_reason": (
                "The existing locked experiments store RGB checkpoint and history hashes, "
                "not the actual historical RGB sequence. Video and hashes are not equivalent "
                "to the model's selected history images."),
        })
        labels.append({
            "key": list(key), "context_id": safe, "split": "exploratory_development_only",
            "Y00_release": outcomes["00"], "Y10_right90": outcomes["10"],
            "Y01_history_reset": outcomes["01"], "Y11_right90_reset": outcomes["11"],
            "interaction_gamma": gamma,
            "delta_success_right90": outcomes["10"] - outcomes["00"],
            "delta_success_history_reset": outcomes["01"] - outcomes["00"],
            "delta_success_right90_reset": outcomes["11"] - outcomes["00"],
            "any_candidate_success": bool(any(outcomes.values())),
            "extra_actions": costs,
        })
        for code, (arm, physical_factor, memory_factor) in ARMS.items():
            long_rows.append({
                "context_id": safe, "scene_id": scene, "episode_id": episode,
                "split": "exploratory_development_only", "arm": arm, "arm_code": code,
                "physical_factor": physical_factor, "memory_factor": memory_factor,
                **metrics[code],
                "external_recovery_actions": costs[code],
                "delta_success_vs_release": outcomes[code] - outcomes["00"],
                "interaction_gamma": gamma,
            })
        checked.append((cp_image, trace[:prefix_count], safe))
    # Only now create output: all four paired outcome sources were validated.
    out_root.mkdir(parents=True, exist_ok=False)
    for cp_image, prefix, safe in checked:
        directory = out_root / "contexts" / safe
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(cp_image, directory / "current_rgb.png")
        # The raw source trace has privileged positions. Export action IDs only.
        write_jsonl(directory / "prefix_actions.jsonl",
                    ({"native_action_index": i, "action": int(r["action"])}
                     for i, r in enumerate(prefix)))
    write_jsonl(out_root / "contexts.jsonl", contexts)
    write_jsonl(out_root / "paired_labels.jsonl", labels)
    with (out_root / "candidate_rows.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    counts = {str(g): sum(r["interaction_gamma"] == g for r in labels)
              for g in (-2, -1, 0, 1, 2)}
    metrics_by_arm = {}
    for code, (arm, _, _) in ARMS.items():
        arm_rows = [r for r in long_rows if r["arm_code"] == code]
        metrics_by_arm[arm] = {
            "success": sum(r["success"] for r in arm_rows),
            "rescued": sum(r["delta_success_vs_release"] == 1 for r in arm_rows),
            "harmed": sum(r["delta_success_vs_release"] == -1 for r in arm_rows),
            "total_external_actions": sum(r["external_recovery_actions"] for r in arm_rows),
        }
    report = {
        "schema": "recovery_paired_dataset_report.v1",
        "status": "PAIRED_LABELS_READY_HISTORY_RGB_NOT_CAPTURED",
        "checkpoint_count": len(labels), "candidate_count": len(long_rows),
        "arm_order": [v[0] for v in ARMS.values()],
        "interaction_gamma_counts": counts, "metrics_by_arm": metrics_by_arm,
        "history_rgb_available": False,
        "ready_for_full_multimodal_training": False,
        "data_use": "exploratory_development_only",
        "source_runs": {
            "checkpoint": str(checkpoint_root),
            "turn": str(turn_root),
            "candidates": str(candidate_root),
        },
        "notes": [
            "No independent train/validation/test split is created from the six inspected cases.",
            "Ground-truth pose and distance are excluded from selector feature exports.",
            "The external turn cost is 6 native body actions; main-loop steps are not physical-action counts.",
            "Gamma is an observed paired outcome difference, not a proven population causal effect.",
            "Missing historical RGB must be collected separately at a future frozen-policy replay.",
        ],
    }
    write_json(out_root / "dataset_report.json", report)
    evidence = []
    input_files = [
        checkpoint_root / "checkpoint_report.json",
        checkpoint_root / "control/progress.json",
        checkpoint_root / "control/actions.jsonl",
        turn_root / "turn_report.json", turn_root / "turn_protocol.json",
        turn_root / "right90/progress.json",
        candidate_root / "candidate_report.json",
        candidate_root / "candidate_protocol.json",
        candidate_root / "history_reset/progress.json",
        candidate_root / "right90_reset/progress.json",
    ]
    evidence += [{"path": str(p), "sha256": digest(p)} for p in input_files]
    write_json(out_root / "input_provenance.json", {
        "schema": "recovery_paired_input_provenance.v1",
        "source_files": evidence,
        "export_files": [
            {"relative_path": str(p.relative_to(out_root)), "sha256": digest(p)}
            for p in sorted(out_root.rglob("*")) if p.is_file()
        ],
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-run", default="data/orientation_checkpoint_01")
    parser.add_argument("--turn-run", default="data/orientation_turn_01")
    parser.add_argument("--candidate-run", default="data/recovery_candidates_01")
    parser.add_argument("--out-dir", default="data/recovery_paired_dataset_01")
    args = parser.parse_args()
    report = make_dataset(args.checkpoint_run, args.turn_run,
                          args.candidate_run, args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
