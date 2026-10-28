"""Compare predeclared physical-angle x history effects using locked native runs.

No GPU. Default: 3x2 existing arms (0, +/-90) from six development cases.
With --angle45-run: 5x2 (0, +/-45, +/-90) with exact same six checkpoints.
All physical actions use real native 15-degree turns; no oracle angle selection.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_dataset_builder as dataset

ANGLES_BASE = (-90, 0, 90)
ANGLES_FULL = (-90, -45, 0, 45, 90)
SCHEMA = "recovery_multiview_comparison.v1"


def effects(values: dict[int, dict[int, int]]):
    """Potential outcomes indexed [degrees][history_reset] for one checkpoint."""
    if 0 not in values or set(values[0]) != {0, 1}:
        raise ValueError("Required matched RELEASE and HISTORY RESET controls")
    y00, y01 = values[0][0], values[0][1]
    result = {}
    for angle, pair in values.items():
        if set(pair) != {0, 1} or any(y not in (0, 1) for y in pair.values()):
            raise ValueError("Incomplete/nonbinary physical-memory factor pair")
        result[angle] = {
            "Y_keep": pair[0], "Y_reset": pair[1],
            "gamma_vs_no_turn": (pair[1] - pair[0] - y01 + y00) if angle else 0,
            "relative_keep": pair[0] - y00,
            "relative_reset": pair[1] - y00,
            "joint_only_rescue": bool(y00 == 0 and y01 == 0 and pair[0] == 0 and pair[1] == 1),
        }
    return result


def assert_true(x, msg):
    if not x:
        raise ValueError("NOT_COMPARABLE: " + msg)


def key(row):
    return tuple(str(x) for x in row["key"])


def per_progress(path, expected):
    return dataset.indexed_progress(path, expected)


def paired_boundary(run, arm, case):
    folder = run / arm / "checkpoints" / ("_".join(case["key"]))
    dataset.check_boundary(folder / "boundary.json", case["checkpoint"], arm)
    return folder


def read_completed(report_path, schema):
    rep = dataset.load_json(report_path)
    assert_true(rep.get("schema") == schema and rep.get("status") == "READY_FOR_INTERPRETATION",
                str(report_path) + " incomplete or schema differs")
    return rep


def compare(source_cp: Path, original_turn: Path, previous: Path, angle45: Path | None = None):
    """Validate input identity, native prefixes and outcomes before extracting labels."""
    cp = source_cp.resolve()
    original_turn = original_turn.resolve()
    previous = previous.resolve()
    cases, progress, turn_data, cand_data, turn_proto, cand_proto = (
        dataset.verify_input_structure(cp, original_turn, previous)
    )
    expected = {key(c) for c in cases}
    turn_report = read_completed(original_turn / "turn_report.json",
                                 "orientation_turn_report.v1")
    cand_report = read_completed(previous / "candidate_report.json",
                                 "recovery_candidates_report.v1")
    base_trace = dataset.group_action_traces(cp / "control/actions.jsonl", expected)
    runs = {(0, 0): (cp, "control", progress["00"]),
            (0, 1): (previous, "history_reset", progress["01"]),
            (-90, 0): (original_turn, "right90", progress["10"]),
            (-90, 1): (previous, "right90_reset", progress["11"])}
    for deg, name, reset_arm in ((90, "left90", "left90_reset"),):
        idx = dataset.index_rows(turn_report["outcomes"][name], name, expected)
        p = per_progress(original_turn / name / "progress.json", expected)
        c_idx = dataset.index_rows(
            [r for r in cand_report["outcomes"] if r["arm"] == reset_arm],
            reset_arm, expected)
        cr = per_progress(previous / reset_arm / "progress.json", expected)
        for case in cases:
            k = key(case)
            assert_true(float(idx[k]["success"]) == float(p[k]["success"]),
                        "left90 source report/progress")
            assert_true(float(c_idx[k]["success"]) == float(cr[k]["success"]),
                        "left90_reset report/progress")
            assert_true(int(c_idx[k]["external_actions"]) == 6, "left90_reset turn actions")
        runs[(deg, 0)] = (original_turn, name, p)
        runs[(deg, 1)] = (previous, reset_arm, cr)

    new_protocol = None
    if angle45 is not None:
        angle45 = angle45.resolve()
        new_protocol = dataset.load_json(angle45 / "candidate_protocol.json")
        assert_true(new_protocol.get("schema") == "recovery_multiview_45.v1",
                    "unexpected 45-degree protocol schema")
        assert_true(new_protocol.get("cases") == cases,
                    "45-degree checkpoints differ")
        assert_true(new_protocol.get("arms") ==
                    ["left45_keep", "right45_keep", "left45_reset", "right45_reset"],
                    "45-degree arms changed")
        assert_true(new_protocol.get("physical_grid", {}).get("offsets") == {
                    "left45_keep": 45, "right45_keep": -45,
                    "left45_reset": 45, "right45_reset": -45},
                    "45-degree angle declaration changed")
        assert_true(Path(dataset.load_json(
            angle45 / "manifest.json")["recovery_candidates"]["source_run"]).resolve() == cp,
                    "45-degree run points to another source")
        newer = read_completed(angle45 / "candidate_report.json",
                               "recovery_multiview_45_report.v1")
        for deg, prefix in ((45, "left45"), (-45, "right45")):
            for reset, suffix in ((0, "keep"), (1, "reset")):
                name = prefix + "_" + suffix
                idx = dataset.index_rows(
                    [r for r in newer["outcomes"] if r["arm"] == name],
                    name, expected)
                p = per_progress(angle45 / name / "progress.json", expected)
                for case in cases:
                    k = key(case)
                    assert_true(float(idx[k]["success"]) == float(p[k]["success"]),
                                name + " report/progress SR")
                    assert_true(int(idx[k]["external_actions"]) == 3,
                                name + " must contain 3 physical actions")
                    assert_true(int(idx[k]["history_prompt_events"]) > 0 if reset
                                else int(idx[k]["history_prompt_events"]) == 0,
                                name + " history rule differs")
                runs[(deg, reset)] = (angle45, name, p)

    angles = ANGLES_FULL if angle45 is not None else ANGLES_BASE
    assert_true(set(runs) == {(a, h) for a in angles for h in (0, 1)},
                "factor grid incomplete")
    # Original control boundary and all arm action traces must match the native prefix.
    for (deg, h), (root, arm, _) in runs.items():
        if (deg, h) == (0, 0):
            continue
        tagged = dataset.group_action_traces(root / arm / "turn_actions.jsonl", expected)
        for case in cases:
            k = key(case)
            if root == original_turn:
                paired_boundary(root, arm, case)
            else:
                paired_boundary(root, arm, case)
            dataset.verify_action_prefix(
                base_trace[k], tagged[k], k,
                case["checkpoint"]["prefix_action_count"], arm)
    pairs = []
    for case in cases:
        k = key(case)
        instruction = progress["00"][k]["episode_instruction"]
        values, measures = {}, {}
        for a in angles:
            values[a] = {}
            for m in (0, 1):
                root, arm, p = runs[(a, m)]
                record = p[k]
                assert_true(record["episode_instruction"] == instruction,
                            "instruction mismatch " + repr((k, arm)))
                values[a][m] = int(float(record["success"]))
                measures[str(a) + "_" + str(m)] = {
                    "arm": arm, "SR": values[a][m],
                    "OS": int(float(record["os"])), "NE": float(record["ne"]),
                    "main_steps": int(record["steps"]),
                    "physical_turn_actions": abs(a) // 15,
                }
        pair_effects = effects(values)
        pairs.append({
            "key": list(k),
            "split": "exploratory_development_only",
            "instruction": instruction,
            "factors": {str(a): {str(h): values[a][h] for h in (0, 1)} for a in angles},
            "gamma_by_angle": {str(a): pair_effects[a]["gamma_vs_no_turn"] for a in angles if a},
            "joint_only_rescue_angles": [a for a in angles if a and pair_effects[a]["joint_only_rescue"]],
            "measurements": measures,
        })
    stats = {}
    for a in angles:
        for h in (0, 1):
            vals = [int(p["factors"][str(a)][str(h)]) for p in pairs]
            base = [int(p["factors"]["0"]["0"]) for p in pairs]
            stats[str(a) + "_" + str(h)] = {
                "angle_degrees": a, "history_reset": bool(h),
                "SR": sum(vals), "rescued": sum(b == 0 and y == 1 for b, y in zip(base, vals)),
                "harmed": sum(b == 1 and y == 0 for b, y in zip(base, vals)),
                "per_case_physical_actions": abs(a) // 15,
            }
    return {
        "schema": SCHEMA, "status": "EXPLORATORY_COMPLETED_FACTORIAL",
        "dataset_type": "preselected_six_case_development",
        "angles_deg": angles, "N_checkpoint": len(cases), "N_arms": len(runs),
        "N_rollouts_including_reused_control": len(cases) * len(runs),
        "arm_statistics": stats, "cases": pairs,
        "oracle_upper_bound_success": sum(
            any(int(v) for vals in p["factors"].values() for v in vals.values())
            for p in pairs),
        "limits": [
            "Angles and cases informed by exploratory development; no independent test",
            "Action counts are native 15-degree body turns, not native main-step counts",
            "History reset filters System-2 input history, not model weights",
            "Non-privileged online trigger and trained selector do not yet exist",
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-run", default="data/orientation_checkpoint_01")
    parser.add_argument("--turn-run", default="data/orientation_turn_01")
    parser.add_argument("--candidate-run", default="data/recovery_candidates_01")
    parser.add_argument("--angle45-run")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root = Path(args.out_dir).resolve()
    if root.exists():
        raise FileExistsError("Report output is create-only; use a new directory")
    report = compare(Path(args.checkpoint_run), Path(args.turn_run),
                     Path(args.candidate_run),
                     Path(args.angle45_run) if args.angle45_run else None)
    root.mkdir(parents=True, exist_ok=False)
    dataset.write_json(root / "angle_factorial_report.json", report)
    dataset.write_jsonl(root / "per_checkpoint_angle_matrix.jsonl", report["cases"])
    print(json.dumps({
        "status": report["status"], "angles_deg": report["angles_deg"],
        "N_checkpoint": report["N_checkpoint"], "arm_statistics": report["arm_statistics"],
        "oracle_upper_bound_success": report["oracle_upper_bound_success"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
