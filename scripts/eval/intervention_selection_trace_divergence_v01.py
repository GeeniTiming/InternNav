"""Read-only posthoc action-trace divergence export for audited v0.1 VLN.

Uses the existing audited mechanism review (four outcome-selected TRAIN
episodes). Extracts all 32 real source action traces, excludes *forced*
pre-arm yaw and SEEK scans from learned-policy continuation comparisons,
and records exactly where observed continuation action codes first differ.

CAUTION: unequal action codes are NOT proof of different S2 beliefs or
the cause of navigation recovery. S2 text is separately recorded only for
post-treatment intervention arms; perturbed KEEP's first S2 was not saved.

No GPU, no model, no privileged online features, no rerun, no label changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

SCHEMA = "intervention_selection_trace_divergence.v0.1"
ACTIONS = ("KEEP", "SEEK_LEFT45", "SEEK_RIGHT45", "HISTORY_RESET")
TARGET_SCENES = {
    ("i5noydFURQK", "70"),
    ("r1Q1Z4BcV1o", "106"),
    ("rPc6DW4iMge", "73"),
    ("PX4nDJXEHrG", "79"),
}
OUTPUTS = (
    "trace_divergence_vs_keep.jsonl",
    "continuation_action_traces.jsonl",
    "decision_comparisons.jsonl",
)
SOURCE_OUTPUTS = (
    "review_contexts.jsonl", "review_arm_evidence.jsonl",
    "human_annotation_templates.jsonl",
)
STOP = 0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def read_lines(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def emit_rows(rows):
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def key_of(row):
    return (PurePosixPath(str(row["scene_id"]).replace("\\", "/")).stem,
            str(row["episode_id"]))


def phase_segments(rows, gate_count, mode, action):
    """Separate actual policy continuation from deliberate forced yaw actions.

    Tagged intervention traces use 'alignment', 'external_recovery',
    and 'continuation'; natural KEEP native actions are untagged.
    """
    if type(gate_count) is not int or gate_count < 1 or gate_count >= len(rows):
        raise ValueError("Missing true checkpoint or continuation")
    after = rows[gate_count:]
    if mode == "natural" and action == "KEEP":
        if any(row.get("phase") not in (None, "continuation") for row in after):
            raise ValueError("Unexpected phase in natural native KEEP")
        return {"alignment": [], "external_recovery": [], "continuation": after}
    names = ("alignment", "external_recovery", "continuation")
    parts = {name: [r for r in after if r.get("phase") == name]
             for name in names}
    if sum(map(len, parts.values())) != len(after):
        raise ValueError("Unknown phase after the pre-arm gate")
    phases = [r["phase"] for r in after]
    expected_order = [name for name in names for _ in parts[name]]
    if phases != expected_order:
        raise ValueError("Invalid causal order of forced yaw and continuation")
    if len(parts["alignment"]) != (3 if mode == "perturbed" else 0):
        raise ValueError("Wrong physical perturbation action count")
    if len(parts["external_recovery"]) != (
        6 if action.startswith("SEEK_") else 0
    ):
        raise ValueError("Wrong SEEK physical scan action count")
    if not parts["continuation"]:
        raise ValueError("No autonomous continuation after intervention")
    if action == "SEEK_LEFT45":
        seq = [2, 2, 2, 3, 3, 3]
    elif action == "SEEK_RIGHT45":
        seq = [3, 3, 3, 2, 2, 2]
    else:
        seq = []
    if [int(r["action"]) for r in parts["external_recovery"]] != seq:
        raise ValueError("Physical scan did not follow registered yaw sequence")
    return parts


def compare_prefix(native, other, gate_count):
    """Only compare common pre-gate physical actions/poses, never phases."""
    if gate_count > len(native) or gate_count > len(other):
        raise ValueError("Insufficient prefix actions")
    for i in range(gate_count):
        a, b = native[i], other[i]
        if int(a["action"]) != int(b["action"]):
            raise ValueError(f"Native prefix action changed at {i}")
        if bool(a["done"]) != bool(b["done"]):
            raise ValueError(f"Native prefix done changed at {i}")
        for field in ("position", "orientation_wxyz"):
            av, bv = a[field], b[field]
            if len(av) != len(bv) or any(
                abs(float(x)-float(y)) > 1e-4
                for x, y in zip(av, bv)
            ):
                raise ValueError(f"Native prefix pose changed at {i}:{field}")
    return True


def first_action_divergence(a, b):
    """Compare action sequences at the same *relative continuation index*.

    This is not a time-aligned S2 decision comparison; a differing code
    or trajectory endpoint does not establish a causal mechanism.
    """
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    if i == len(a) == len(b):
        return {
            "first_different_continuation_index": None,
            "reason": "IDENTICAL_ACTION_CODES",
            "shared_action_prefix_length": i,
        }
    if i == min(len(a), len(b)):
        reason = "ONE_SEQUENCE_TERMINATES"
    else:
        reason = "ACTION_CODE_DIFFERS"
    return {
        "first_different_continuation_index": i,
        "reason": reason, "shared_action_prefix_length": i,
        "keep_action_at_divergence": a[i] if i < len(a) else None,
        "treated_action_at_divergence": b[i] if i < len(b) else None,
    }


def require_review(root):
    root = Path(root).resolve(strict=True)
    report = read_json(root / "review_report.json")
    if (report.get("schema") != "intervention_selection_mechanism_review.v0.1"
            or report.get("status") !=
               "POSTHOC_MECHANISM_EVIDENCE_REVIEW_READY_NOT_ANNOTATED"
            or report.get("N_review_scenes") != 4
            or report.get("N_review_contexts") != 8
            or report.get("N_review_action_outcomes") != 32
            or report.get("selection_after_observing_results") is not True):
        raise ValueError("Original fully audited posthoc review required")
    for name in SOURCE_OUTPUTS:
        digest = report.get("output_hashes", {}).get(name)
        if not digest or sha(root / name) != digest:
            raise ValueError("Saved review table changed: " + name)
    # Revalidate SOURCE provenance too, including original reference RGB and
    # source evaluator/manifest code hashes frozen when review was created.
    for src, expected in report["source_hashes"].items():
        if sha(src) != expected:
            raise ValueError("Review source evidence changed: " + src)
    records = read_lines(root / "review_arm_evidence.jsonl")
    contexts = read_lines(root / "review_contexts.jsonl")
    if len(records) != 32 or len(contexts) != 8:
        raise ValueError("Frozen review cardinality changed")
    expected_ids = {r["context_id"] for r in contexts}
    if len(expected_ids) != 8:
        raise ValueError("Repeated review context")
    groups = defaultdict(dict)
    for item in records:
        cid, action = item["context_id"], item["action"]
        key = tuple(item["episode_key"])
        if (cid not in expected_ids or key not in TARGET_SCENES
                or action not in ACTIONS or action in groups[cid]):
            raise ValueError("Unexpected review case or duplicated action")
        if item.get("selected_after_outcomes_observed") is not True:
            raise ValueError("Posthoc selection warning missing")
        groups[cid][action] = item
    if len(groups) != 8 or any(set(d) != set(ACTIONS) for d in groups.values()):
        raise ValueError("Missing same-context real four-arm evidence")
    return root, report, groups


def collect(root):
    src, review_report, groups = require_review(root)
    paths = {Path(v["trace"]["trace_source_path"]).resolve()
             for d in groups.values() for v in d.values()}
    if len(paths) != 8:
        raise ValueError("Expect two KEEP trace sources and six intervention arms")
    cache = {}
    for p in sorted(paths):
        trace = defaultdict(list)
        for row in read_lines(p):
            trace[key_of(row)].append(row)
        cache[p] = trace
    reference = {}
    for cid, arms in groups.items():
        mode = arms["KEEP"]["source_type"]
        k = tuple(arms["KEEP"]["episode_key"])
        if mode not in ("natural", "perturbed") or not cid.endswith("_" + mode):
            raise ValueError("Invalid state identity")
        nat = (cid[:-len("_" + mode)] + "_natural")
        if nat not in groups:
            raise ValueError("Natural scene baseline not present")
        nref = groups[nat]["KEEP"]
        if tuple(nref["episode_key"]) != k:
            raise ValueError("Natural and perturbed episode identities differ")
        natural_trace = cache[
            Path(nref["trace"]["trace_source_path"]).resolve()][k]
        reference[cid] = natural_trace

    all_actions, divergences, decisions = [], [], []
    seen = set()
    for cid, arms in sorted(groups.items()):
        mode = arms["KEEP"]["source_type"]
        key = tuple(arms["KEEP"]["episode_key"])
        gate = arms["KEEP"]["trace"]["prefix_native_action_count"]
        control_seq = None
        first_s2 = {}
        for action in ACTIONS:
            entry = arms[action]
            path = Path(entry["trace"]["trace_source_path"]).resolve()
            rows = cache[path][key]
            if not rows:
                raise ValueError("Missing real action trace for " + cid + action)
            compare_prefix(reference[cid], rows, gate)
            parts = phase_segments(rows, gate, mode, action)
            if len(rows) - gate != entry["trace"]["actions_since_native_gate"]:
                raise ValueError("Saved review action count differs")
            if (len(parts["alignment"]) != entry["trace"]["perturbation_actions"]
                    or len(parts["external_recovery"]) !=
                        entry["trace"]["extra_yaw_actions"]):
                raise ValueError("Physical yaw count differs from review")
            cont = parts["continuation"]
            seq = [int(r["action"]) for r in cont]
            if action == "KEEP":
                control_seq = seq
            first = entry["evidence"]["first_system2"]
            first_s2[action] = first if isinstance(first, dict) else None
            for i, row in enumerate(cont):
                all_actions.append({
                    "schema": SCHEMA, "context_id": cid,
                    "episode_key": list(key), "source_type": mode,
                    "action": action, "continuation_index": i,
                    "action_code": int(row["action"]),
                    "main_step_if_recorded": row.get("main_step"),
                    "physical_position_OFFLINE_ONLY": row["position"],
                    "physical_orientation_OFFLINE_ONLY":
                        row["orientation_wxyz"],
                    "env_done": bool(row["done"]),
                    "ground_truth_trajectory_must_never_be_online_input": True,
                })
            seen.add((cid, action))
            if action == "KEEP":
                continue
            div = first_action_divergence(control_seq, seq)
            divergences.append({
                "schema": SCHEMA, "context_id": cid,
                "episode_key": list(key), "source_type": mode,
                "action": action, "reference": "KEEP",
                "keep_success_OFFLINE_ONLY":
                    arms["KEEP"]["observed_success_OFFLINE_ONLY"],
                "action_success_OFFLINE_ONLY":
                    entry["observed_success_OFFLINE_ONLY"],
                "keep_ne_OFFLINE_ONLY": arms["KEEP"]["observed_ne_OFFLINE_ONLY"],
                "action_ne_OFFLINE_ONLY": entry["observed_ne_OFFLINE_ONLY"],
                "keep_continuation_action_count": len(control_seq),
                "action_continuation_action_count": len(seq),
                "first_32_keep_actions": control_seq[:32],
                "first_32_action_actions": seq[:32],
                "true_scan_actions_removed_from_policy_comparison": True,
                "true_perturbation_actions_removed_from_policy_comparison": True,
                "action_index_not_a_System2_decision_time": True,
                "difference_does_not_identify_recovery_mechanism": True,
                **div,
            })
        for a, b in (("SEEK_LEFT45", "SEEK_RIGHT45"),
                     ("SEEK_RIGHT45", "HISTORY_RESET")):
            aa, bb = first_s2[a], first_s2[b]
            if not aa or not bb or (
                aa.get("status") != "ACTUAL_RECORDED_POST_INTERVENTION"
                or bb.get("status") != "ACTUAL_RECORDED_POST_INTERVENTION"
            ):
                raise ValueError("First post-intervention S2 text missing")
            decisions.append({
                "schema": SCHEMA, "context_id": cid,
                "episode_key": list(key), "source_type": mode,
                "action_a": a, "action_b": b,
                "first_S2_a": aa["text"], "first_S2_b": bb["text"],
                "identical_first_S2_text": aa["text"] == bb["text"],
                "outcome_a_OFFLINE_ONLY":
                    arms[a]["observed_success_OFFLINE_ONLY"],
                "outcome_b_OFFLINE_ONLY":
                    arms[b]["observed_success_OFFLINE_ONLY"],
                "first_S2_text_does_not_explain_full_trajectory": True,
                "KEEP_first_S2_available":
                    first_s2["KEEP"]["text"] is not None,
            })
    if len(seen) != 32 or len(divergences) != 24 or len(decisions) != 16:
        raise ValueError("Incomplete first-action and first-S2 review")
    info = {
        "schema": SCHEMA,
        "status": "POSTHOC_CONTINUATION_TRACES_EXPORTED_NOT_CAUSALLY_IDENTIFIED",
        "original_selection_status": review_report["status"],
        "N_scene_groups": 4, "N_contexts": 8, "N_real_arm_traces": 32,
        "N_effects_vs_KEEP": 24, "N_first_S2_pair_comparisons": 16,
        "N_action_rows": len(all_actions),
        "source_review_dir": str(src),
        "review_report_sha256": sha(src / "review_report.json"),
        "original_trace_sha256": {str(p): sha(p) for p in sorted(paths)},
        "scope": "TRAIN posthoc qualitative study; action divergence by relative policy-continuation action index ONLY, no S2 decision-time alignment, no causal mechanism identification; no online use of GT positions or outcomes",
        "no_model_training": True,
        "no_new_rollouts": True,
        "no_new_selection_dataset": True,
        "no_heldout_efficacy_claim": True,
    }
    return info, {
        OUTPUTS[0]: emit_rows(divergences),
        OUTPUTS[1]: emit_rows(all_actions),
        OUTPUTS[2]: emit_rows(decisions),
    }


def build(review_dir, out_dir):
    dest = Path(out_dir).resolve()
    if dest.exists():
        raise FileExistsError("Trace evidence export is create-only")
    info, data = collect(review_dir)
    dest.mkdir(parents=True, exist_ok=False)
    for name, content in data.items():
        (dest / name).write_bytes(content)
    info["output_hashes"] = {n: sha(dest / n) for n in OUTPUTS}
    (dest / "trace_divergence_report.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n", encoding="utf-8")
    with zipfile.ZipFile(
        dest / "trace_divergence_share.zip", "w", zipfile.ZIP_DEFLATED,
        compresslevel=8,
    ) as f:
        for n in (*OUTPUTS, "trace_divergence_report.json"):
            f.write(dest / n, n)
    print(json.dumps({
        "status": info["status"],
        "N_scene_groups": info["N_scene_groups"],
        "N_contexts": info["N_contexts"],
        "N_real_arm_traces": info["N_real_arm_traces"],
        "N_action_rows": info["N_action_rows"],
        "N_effects_vs_KEEP": info["N_effects_vs_KEEP"],
        "N_first_S2_pair_comparisons": info["N_first_S2_pair_comparisons"],
        "uploadable_zip": str(dest / "trace_divergence_share.zip"),
    }, indent=2, ensure_ascii=False))
    return info


def audit(out_dir):
    dest = Path(out_dir).resolve(strict=True)
    saved = read_json(dest / "trace_divergence_report.json")
    rebuilt, content = collect(saved["source_review_dir"])
    if (rebuilt != {k:v for k,v in saved.items() if k != "output_hashes"}):
        raise ValueError("Rebuilt trace report or source hashes changed")
    for name in OUTPUTS:
        if sha(dest / name) != saved["output_hashes"][name]:
            raise ValueError("Saved trace changed: " + name)
        if hashlib.sha256(content[name]).hexdigest() != saved["output_hashes"][name]:
            raise ValueError("Recomputed divergence differs: " + name)
    with zipfile.ZipFile(dest / "trace_divergence_share.zip") as f:
        if set(f.namelist()) != set((*OUTPUTS, "trace_divergence_report.json")):
            raise ValueError("Unexpected ZIP evidence files")
        for name in (*OUTPUTS, "trace_divergence_report.json"):
            if f.read(name) != (dest / name).read_bytes():
                raise ValueError("Export ZIP differs from saved audited files")
    print("POSTHOC_TRACE_DIVERGENCE_AUDITED",
          saved["N_scene_groups"], "scenes",
          saved["N_contexts"], "contexts",
          saved["N_real_arm_traces"], "arms",
          saved["N_action_rows"], "policy action rows")
    return saved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    ss = p.add_subparsers(dest="command", required=True)
    x = ss.add_parser("build")
    x.add_argument("--review-dir", required=True)
    x.add_argument("--out-dir", required=True)
    x.set_defaults(fn=lambda v: build(v.review_dir, v.out_dir))
    x = ss.add_parser("audit")
    x.add_argument("--out-dir", required=True)
    x.set_defaults(fn=lambda v: audit(v.out_dir))
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
