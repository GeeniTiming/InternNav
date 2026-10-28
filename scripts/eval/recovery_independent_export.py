"""Export paired 5x2 recovery supervision and real frozen-model RGB history.

Only uses already completed, checked native CONTROL and all nine interventions.
Produces a self-contained acquisition dataset with non-privileged selector inputs;
does not perform any train/test split or silently promote dev examples to test.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_independent_factorial as factorial

pilot, physical, capture = factorial.pilot, factorial.physical, factorial.capture
ARMS = ("release",) + factorial.ARMS
ANGLES = {"release": 0, "history_reset": 0,
          **factorial.ROTATIONS}


def export(checkpoint_run, factorial_run, out_dir):
    checkpoint_run = Path(checkpoint_run).resolve()
    factorial_run = Path(factorial_run).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError("Create-only dataset: " + str(out_dir))
    # read_run and source_check independently verify full native control hashes,
    # matched checkpoint replay and the original weight/config identity.
    _, source, protocol = factorial.read_run(factorial_run)
    if source.resolve() != checkpoint_run:
        raise ValueError("The factorial references another checkpoint source")
    source_report = pilot.read_json(checkpoint_run / "checkpoint_report.json")
    treatment_report = pilot.read_json(factorial_run / "candidate_report.json")
    if (source_report["schema"] != "independent_recovery_checkpoints.v1"
            or source_report["status"] != "CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION"
            or treatment_report["schema"] != "recovery_independent_factorial_report.v1"
            or treatment_report["status"] != "READY_FOR_INTERPRETATION"):
        raise ValueError("Source experiments incomplete or from a different version")
    cases = source_report["cases"]
    if len(cases) != len(protocol["cases"]) or any(
            a["key"] != b["key"] or a["checkpoint"] != b["checkpoint"]
            for a, b in zip(cases, protocol["cases"])):
        raise ValueError("Cases/checkpoints changed")
    keys = {tuple(c["key"]) for c in cases}
    labels = {}
    for row in treatment_report["outcomes"]:
        key = (tuple(row["key"]), row["arm"])
        if key in labels or key[0] not in keys or row["arm"] not in factorial.ARMS:
            raise ValueError("Duplicate/unknown candidate outcome " + repr(key))
        if not row["eligible"] or int(row["external_actions"]) != (
                abs(ANGLES[row["arm"]]) // 15):
            raise ValueError("Ineligible or incorrectly charged candidate")
        if int(row["success"]) not in (0, 1):
            raise ValueError("Nonbinary candidate success")
        labels[key] = row
    if len(labels) != len(cases) * len(factorial.ARMS):
        raise ValueError("Incomplete 5x2 outcome labels")

    progress = capture.read_progress(checkpoint_run / "control/progress.json")
    if set(progress) != keys:
        raise ValueError("Native RELEASE progress cohort mismatch")
    full_traces = {}
    for row in physical.read_lines(checkpoint_run / "control/actions.jsonl"):
        full_traces.setdefault(capture.episode_key(row), []).append(row)
    if set(full_traces) != keys:
        raise ValueError("Native action records incomplete")

    feature_rows, paired_rows, candidates, safe_sources = [], [], [], []
    all_scenes = set()
    for case in cases:
        k = tuple(case["key"])
        scene, ep = k
        if scene in all_scenes:
            raise ValueError("New acquisition must contain one route group per scene")
        all_scenes.add(scene)
        gate = case["checkpoint"]
        if case["status"] != "COMPARABLE_CHECKPOINT_CAPTURED" or gate is None:
            raise ValueError("Missing observed online-only checkpoint " + repr(k))
        folder = checkpoint_run / "control/checkpoints" / (scene + "_" + ep)
        data = pilot.read_json(folder / "context_inputs.json")
        if (data["schema"] != "independent_recovery_context_inputs.v1"
                or data["key"] != list(k)
                or int(data["main_step"]) != int(gate["step"])
                or data["history_sha256"] != gate["history_sha256"]):
            raise ValueError("Saved RGB history does not match checkpoint")
        sampled = data["sampled_history"]
        if len(sampled) != len({int(t["native_history_index"]) for t in sampled}):
            raise ValueError("Duplicate sampled historical index")
        to_copy = [("current_model_rgb.png", data["current_model_rgb_png_sha256"])]
        for frame in sampled:
            idx = int(frame["native_history_index"])
            if idx < 0 or idx >= int(data["history_length"]):
                raise ValueError("Invalid historical frame index")
            to_copy.append((frame["path"], frame["png_sha256"]))
        for name, expected in to_copy:
            path = folder / name
            if not path.is_file() or pilot.sha256(path) != expected:
                raise ValueError("Historical RGB file is missing or has changed " + str(path))
        with Image.open(folder / "current_model_rgb.png") as image:
            if physical.history_hash([image.copy()]) != data["current_model_rgb_pixel_sha256"]:
                raise ValueError("Checkpoint current model RGB pixel hash differs")
        for item in sampled:
            with Image.open(folder / item["path"]) as image:
                if physical.history_hash([image.copy()]) != item["pixel_sha256"]:
                    raise ValueError("Historical RGB pixel hash differs")

        actions = full_traces[k][:int(gate["prefix_action_count"])]
        if len(actions) != int(gate["prefix_action_count"]):
            raise ValueError("Recorded actions are shorter than checkpoint prefix")
        safe_id = scene + "_" + ep
        y = {"release": int(progress[k]["success"])}
        for arm in factorial.ARMS:
            y[arm] = int(labels[(k, arm)]["success"])
        if any(v not in (0, 1) for v in y.values()):
            raise ValueError("Invalid native success label")
        factors = {}
        for angle in (-90, -45, 0, 45, 90):
            pair = {}
            for h in (0, 1):
                if angle == 0:
                    arm = "history_reset" if h else "release"
                else:
                    prefix = ("left" if angle > 0 else "right") + str(abs(angle))
                    arm = prefix + ("_reset" if h else "_keep")
                pair[str(h)] = y[arm]
                candidates.append({
                    "context_id": safe_id, "scene_id": scene, "episode_id": ep,
                    "split": "acquisition_development_only",
                    "arm": arm, "turn_angle_deg": angle, "reset_history": h,
                    "success": y[arm],
                    "delta_success_vs_release": y[arm] - y["release"],
                    "recovery_turn_actions": abs(angle) // 15,
                })
            factors[str(angle)] = pair
        gammas = {}
        for angle in (-90, -45, 45, 90):
            pair, no = factors[str(angle)], factors["0"]
            gammas[str(angle)] = pair["1"] - pair["0"] - no["1"] + no["0"]
        paired_rows.append({"context_id": safe_id, "key": list(k), "factors": factors,
                            "gamma_by_angle": gammas,
                            "has_successful_option": bool(any(y.values())),
                            "split": "acquisition_development_only"})
        feature_rows.append({
            "context_id": safe_id, "key": list(k),
            "split": "acquisition_development_only",
            "instruction": progress[k]["episode_instruction"],
            "current_rgb": "contexts/" + safe_id + "/current_model_rgb.png",
            "history_rgb": [
                {"native_index": int(v["native_history_index"]),
                 "relative_path": "contexts/" + safe_id + "/" + v["path"]}
                for v in sampled],
            "prefix_actions": "contexts/" + safe_id + "/prefix_actions.jsonl",
            "remaining_native_main_loop_budget_proxy":
                int(protocol["native_main_step_limit"]) - int(gate["step"]),
            "checkpoint_main_step": int(gate["step"]),
            "no_privileged_selector_features": True,
        })
        safe_sources.append((folder, safe_id, to_copy, actions))

    # Everything validated before creating any output.
    out_dir.mkdir(parents=True, exist_ok=False)
    for folder, safe_id, to_copy, actions in safe_sources:
        dest = out_dir / "contexts" / safe_id
        dest.mkdir(parents=True, exist_ok=False)
        for name, _ in to_copy:
            dst = dest / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / name, dst)
        with (dest / "prefix_actions.jsonl").open("x", encoding="utf-8") as f:
            for i, r in enumerate(actions):
                f.write(json.dumps({"index": i, "action": int(r["action"])}) + "\n")
    with (out_dir / "contexts.jsonl").open("x", encoding="utf-8") as f:
        for r in feature_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "paired_labels.jsonl").open("x", encoding="utf-8") as f:
        for r in paired_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "candidate_rows.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    result = {
        "schema": "independent_recovery_dataset.v1",
        "status": "PAIRED_LABELS_WITH_ACTUAL_NATIVE_HISTORY_RGB",
        "N_contexts": len(feature_rows), "N_candidates": len(candidates),
        "N_unique_scenes": len(all_scenes),
        "history_rgb_available": True,
        "feature_format_complete": True,
        "generalization_evidence": False,
        "data_use": "acquisition_development_only",
        "source_checkpoint_run": str(checkpoint_run),
        "source_factorial_run": str(factorial_run),
        "notes": [
            "No goal/GPS/depth/pathfinder truth exported as selector features.",
            "Images are exact sampled pre-decision native resized RGB images.",
            "No train, validation or test split has been made.",
            "One context supplies ten correlated candidate labels; they are not independent.",
        ],
    }
    pilot.write_json(out_dir / "dataset_report.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-run", required=True)
    parser.add_argument("--factorial-run", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    export(args.checkpoint_run, args.factorial_run, args.out_dir)


if __name__ == "__main__":
    main()
