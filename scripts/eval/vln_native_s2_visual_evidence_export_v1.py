#!/usr/bin/env python3
"""Export selected RGB evidence and exact S2 records from completed native runs.

CPU-only and read-only. Reads paired_protocol.json to locate the frozen native
capture directories, validates global action offsets against control/actions.jsonl,
verifies every PNG against the recorded pixel SHA256, and exports a small,
post-hoc evidence bundle. It never runs the model or modifies source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

DEFAULT_BASE = Path("/mnt/data8t/geenitiming/internnav_data")
SCHEMA = "native_s2_visual_evidence_export.v1"

# E9uDoFAP3SH/64 is a DEVELOPMENT success control, not a validation episode.
TARGETS = (
    {
        "cohort": "development",
        "paired_folder": "intervention_v01_paired_all10_01",
        "key": ("VLzqgDo317F", "58"),
        "role": "native_failure_stagnation",
        "window": (193, 2671),
        "event_start": 257, "event_end": 2671,
        "primary": {"call_type": "ordinary", "model_output": "↓", "minimum_run": 50,
                    "sample_indices": (0, 25, 51)},
        "secondary": {"call_type": "lookdown_followup", "model_output": "371 170",
                      "minimum_run": 29, "sample_indices": (0, 14, 28)},
        "special_outputs": ("378 170",),
    },
    {
        "cohort": "development",
        "paired_folder": "intervention_v01_paired_all10_01",
        "key": ("i5noydFURQK", "70"),
        "role": "native_failure_near_goal_departure",
        "window": (209, 875),
        "event_start": 209, "event_end": 875,
        "primary": {"call_type": "ordinary", "model_output": "←←←←", "minimum_run": 32,
                    "sample_indices": (0, 1, 5, 6, 15, 31)},
        "secondary": None,
        "special_outputs": (),
    },
    {
        "cohort": "development",
        "paired_folder": "intervention_v01_paired_all10_01",
        "key": ("E9uDoFAP3SH", "64"),
        "role": "native_success_lookdown_control",
        "window": (0, 524),
        "primary": {"call_type": "ordinary", "model_output": "↓", "minimum_run": 10,
                    "sample_indices": (0, 4, 9)},
        "secondary": None,
        "special_outputs": (),
    },
    {
        "cohort": "validation",
        "paired_folder": "h1_validation_paired_all11_01",
        "key": ("82sE5b5pLXE", "22"),
        "role": "scene_disjoint_success_turning_control",
        "window": (204, 380),
        "event_start": 204, "event_end": 380,
        "primary": {"call_type": "ordinary", "model_output": "←←←←", "minimum_run": 5,
                    "sample_indices": (0, 2, 4)},
        "secondary": None,
        "special_outputs": (),
    },
)


def canonical_key(scene_id, episode_id):
    scene = PurePosixPath(str(scene_id).replace("\\", "/")).stem
    return scene, str(episode_id)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("{}:{} invalid JSON".format(path, line_no)) from exc


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_digest(image):
    """Exactly match recovery_task_evidence_acquire.pixel_digest()."""
    key = image.mode.encode("ascii") + b"\0" + str(image.size).encode("ascii")
    return hashlib.sha256(key + b"\0" + image.tobytes()).hexdigest()


def group_action_rows(rows):
    """Return per-episode rows and exact global start offsets from file order."""
    grouped = {}
    offsets = {}
    order = []
    closed = set()
    previous = None
    total = 0
    for row in rows:
        key = canonical_key(row["scene_id"], row["episode_id"])
        if key != previous:
            if key in closed:
                raise ValueError("Native action rows are not episode-contiguous: " + repr(key))
            if previous is not None:
                closed.add(previous)
            previous = key
            grouped[key] = []
            offsets[key] = total
            order.append(key)
        grouped[key].append(row)
        total += 1
    return grouped, offsets, total, order


def localize_call_records(rows, offsets, lengths, total_actions):
    """Preserve global offsets and add exact episode-local offsets."""
    localized = defaultdict(list)
    previous_call_id = -1
    previous_global = -1
    for source in rows:
        call_id = int(source["call_id"])
        global_prefix = int(source["native_action_prefix_count"])
        if call_id <= previous_call_id:
            raise ValueError("S2 call_id is not strictly increasing")
        if global_prefix < previous_global:
            raise ValueError("S2 global action-prefix counter decreases")
        previous_call_id = call_id
        previous_global = global_prefix

        key = tuple(str(x) for x in source["episode_key"])
        if key not in offsets or key not in lengths:
            raise ValueError("S2 call references missing native episode: " + repr(key))
        if not 0 <= global_prefix <= total_actions:
            raise ValueError("S2 global prefix outside full action trace: " + repr(source))
        local_prefix = global_prefix - int(offsets[key])
        if not 0 <= local_prefix <= int(lengths[key]):
            raise ValueError(
                "S2 global prefix does not fall inside its episode block: "
                + repr(key) + " global=" + str(global_prefix)
                + " interval=[" + str(offsets[key]) + ","
                + str(offsets[key] + lengths[key]) + "]"
            )
        row = dict(source)
        row["action_prefix_count_global"] = global_prefix
        row["action_prefix_count_episode"] = local_prefix
        validate_call_record(row)
        localized[key].append(row)

    for key, calls in localized.items():
        calls.sort(key=lambda x: (int(x["action_prefix_count_episode"]), int(x["call_id"])))
        steps = [int(x["main_step"]) for x in calls]
        if any(b < a for a, b in zip(steps, steps[1:])):
            raise ValueError("S2 main_step decreases within episode: " + repr(key))
        offsets_local = [int(x["action_prefix_count_episode"]) for x in calls]
        if any(b < a for a, b in zip(offsets_local, offsets_local[1:])):
            raise ValueError("S2 local action offsets decrease within episode: " + repr(key))
    return dict(localized)


def validate_call_record(row):
    if row.get("schema") != "task_evidence_decisions.v1":
        raise ValueError("Unexpected S2 record schema")
    if row.get("call_type") not in ("ordinary", "lookdown_followup"):
        raise ValueError("Unexpected S2 call_type")
    if not isinstance(row.get("images"), list) or not row["images"]:
        raise ValueError("S2 call has no image references")
    if not row.get("model_output") and row.get("model_output") != "":
        raise ValueError("S2 call is missing model_output")
    if not isinstance(row.get("native_parsed_actions"), list):
        raise ValueError("S2 call is missing native_parsed_actions")
    rendered = row.get("rendered_prompt")
    expected = row.get("rendered_prompt_sha256")
    if not isinstance(rendered, str) or hashlib.sha256(rendered.encode("utf-8")).hexdigest() != expected:
        raise ValueError("Rendered prompt SHA256 mismatch for call " + str(row.get("call_id")))
    last_path = row["images"][-1].get("path")
    if row["call_type"] == "ordinary":
        if row.get("current_image_ref") != last_path:
            raise ValueError("Ordinary S2 current_image_ref does not match final image")
    elif row.get("current_image_ref") is not None:
        raise ValueError("Lookdown follow-up unexpectedly has current_image_ref")


def longest_output_run(calls, call_type, model_output, start, end):
    """Longest same-type/same-output run; other call types are interleaved."""
    typed = [
        c for c in calls
        if c["call_type"] == call_type
        and start <= int(c["action_prefix_count_episode"]) <= end
    ]
    runs = []
    current = []
    for call in typed:
        if call.get("model_output") == model_output:
            current.append(call)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)
    if not runs:
        return []
    return max(runs, key=lambda run: (len(run), -int(run[0]["call_id"])))


def add_selected(selected, call, reason):
    call_id = int(call["call_id"])
    if call_id not in selected:
        selected[call_id] = {"call": call, "reasons": []}
    if reason not in selected[call_id]["reasons"]:
        selected[call_id]["reasons"].append(reason)


def sample_run(run, indices):
    if not run:
        raise ValueError("Cannot sample an empty S2 output run")
    result = []
    for index in indices:
        if index < 0 or index >= len(run):
            raise ValueError(
                "Configured sample index {} outside run length {}".format(index, len(run)))
        result.append(run[index])
    return result


def select_target_calls(all_calls, spec):
    key = tuple(spec["key"])
    calls = sorted(all_calls.get(key, []),
                   key=lambda x: (int(x["action_prefix_count_episode"]), int(x["call_id"])))
    if not calls:
        raise ValueError("No S2 records for target episode " + repr(key))
    start, end = spec["window"]
    primary_spec = spec["primary"]
    primary_run = longest_output_run(
        calls, primary_spec["call_type"], primary_spec["model_output"], start, end)
    if len(primary_run) < int(primary_spec["minimum_run"]):
        raise ValueError(
            "Primary repeated-output run too short for {}: expected at least {}, got {}"
            .format(key, primary_spec["minimum_run"], len(primary_run)))

    selected = {}
    for call in sample_run(primary_run, primary_spec["sample_indices"]):
        add_selected(selected, call, "primary_run_sample")

    run_first = primary_run[0]
    run_last = primary_run[-1]
    # Include the immediately preceding S2 call and two calls after the repeated run.
    before = [c for c in calls if int(c["call_id"]) < int(run_first["call_id"])]
    for call in before[-1:]:
        add_selected(selected, call, "pre_run_context")
    after = [c for c in calls if int(c["call_id"]) > int(run_last["call_id"])]
    for call in after[:2]:
        add_selected(selected, call, "post_run_transition")

    # Anchor the evidence to the independently audited native action interval.
    # These are retrospective export anchors, never online detection features.
    for boundary_name in ("event_start", "event_end"):
        boundary = spec.get(boundary_name)
        if boundary is None:
            continue
        before_boundary = [
            c for c in calls
            if int(c["action_prefix_count_episode"]) <= int(boundary)
        ]
        after_boundary = [
            c for c in calls
            if int(c["action_prefix_count_episode"]) >= int(boundary)
        ]
        if before_boundary:
            nearest = before_boundary[-1]
            add_selected(selected, nearest, boundary_name + "_nearest_before")
        if after_boundary:
            nearest = after_boundary[0]
            add_selected(selected, nearest, boundary_name + "_nearest_after")

    secondary_spec = spec.get("secondary")
    secondary_run = []
    if secondary_spec:
        secondary_run = longest_output_run(
            calls, secondary_spec["call_type"], secondary_spec["model_output"], start, end)
        if len(secondary_run) < int(secondary_spec["minimum_run"]):
            raise ValueError(
                "Secondary repeated-output run too short for {}: expected at least {}, got {}"
                .format(key, secondary_spec["minimum_run"], len(secondary_run)))
        for call in sample_run(secondary_run, secondary_spec["sample_indices"]):
            add_selected(selected, call, "secondary_run_sample")

    for output in spec.get("special_outputs", ()):
        matches = [c for c in calls if c.get("model_output") == output
                   and start <= int(c["action_prefix_count_episode"]) <= end]
        for call in matches:
            add_selected(selected, call, "special_output_transition")

    # Include STOP if emitted; for the non-terminating failure, retain the final call.
    stop_calls = [c for c in calls if str(c.get("model_output", "")).strip() == "STOP"]
    if stop_calls:
        add_selected(selected, stop_calls[-1], "terminal_stop_decision")
    else:
        add_selected(selected, calls[-1], "last_recorded_decision")

    # Pair selected LOOK_DOWN calls with their same-main-step follow-up, and vice versa.
    by_step = defaultdict(list)
    for call in calls:
        by_step[int(call["main_step"])].append(call)
    for call in list(x["call"] for x in selected.values()):
        for partner in by_step[int(call["main_step"])]:
            if partner["call_type"] != call["call_type"]:
                add_selected(selected, partner, "paired_same_main_step")

    chosen = sorted(selected.values(),
                    key=lambda x: (int(x["call"]["action_prefix_count_episode"]),
                                   int(x["call"]["call_id"])))
    return chosen, {
        "primary_run_length": len(primary_run),
        "primary_run_call_ids": [int(x["call_id"]) for x in primary_run],
        "secondary_run_length": len(secondary_run),
        "secondary_run_call_ids": [int(x["call_id"]) for x in secondary_run],
        "selected_call_count": len(chosen),
    }


def verify_source_image(capture_root, ref):
    digest = str(ref.get("pixel_sha256", ""))
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("Malformed pixel_sha256 in image reference")
    rel = PurePosixPath(str(ref.get("path", "")))
    if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != "images":
        raise ValueError("Unsafe or unexpected capture image path: " + str(rel))
    if rel.name != digest + ".png":
        raise ValueError("Image filename does not match recorded pixel digest: " + str(rel))
    source = (Path(capture_root) / Path(*rel.parts)).resolve()
    expected_root = (Path(capture_root) / "images").resolve()
    if source.parent != expected_root:
        raise ValueError("Capture image resolves outside images directory")
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(str(source)) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError("Captured evidence image is not PNG: " + str(source))
        actual_digest = pixel_digest(image)
        if actual_digest != digest:
            raise ValueError("Pixel SHA256 mismatch: " + str(source))
        if list(image.size) != [int(ref["width"]), int(ref["height"])]:
            raise ValueError("Image dimensions differ from S2 record: " + str(source))
        if image.mode != str(ref["mode"]):
            raise ValueError("Image mode differs from S2 record: " + str(source))
    return source, sha256_file(source)


def load_cohort(base, cohort, paired_folder):
    paired_root = (Path(base) / paired_folder).resolve()
    protocol_path = paired_root / "paired_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = read_json(protocol_path)
    if not protocol.get("source_native_run"):
        raise ValueError("paired_protocol.json lacks source_native_run: " + str(protocol_path))
    native = Path(protocol["source_native_run"])
    if not native.is_absolute():
        native = (paired_root / native).resolve()
    if not native.is_dir():
        raise FileNotFoundError("Native run directory missing: " + str(native))
    actions_path = native / "control" / "actions.jsonl"
    decisions_path = native / "capture" / "online_decisions.jsonl"
    capture_root = native / "capture"
    for path in (actions_path, decisions_path, capture_root / "images"):
        if not path.exists():
            raise FileNotFoundError(path)
    action_rows = list(read_jsonl(actions_path))
    grouped, offsets, total, order = group_action_rows(action_rows)
    lengths = {key: len(rows) for key, rows in grouped.items()}
    decision_rows = list(read_jsonl(decisions_path))
    localized = localize_call_records(decision_rows, offsets, lengths, total)
    return {
        "cohort": cohort,
        "paired_root": paired_root,
        "protocol_path": protocol_path,
        "native_root": native,
        "actions_path": actions_path,
        "decisions_path": decisions_path,
        "capture_root": capture_root,
        "actions_sha256": sha256_file(actions_path),
        "decisions_sha256": sha256_file(decisions_path),
        "action_total": total,
        "episode_order": order,
        "action_lengths": lengths,
        "action_offsets": offsets,
        "calls_by_episode": localized,
    }


def jsonl_write(path, rows):
    with Path(path).open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def export(args):
    base = Path(args.base).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Create-only output already exists: " + str(output))

    cohort_cache = {}
    target_plans = []
    prepared_images = []
    for spec in TARGETS:
        cohort = spec["cohort"]
        if cohort not in cohort_cache:
            cohort_cache[cohort] = load_cohort(base, cohort, spec["paired_folder"])
        source = cohort_cache[cohort]
        key = tuple(spec["key"])
        if key not in source["action_lengths"]:
            raise ValueError("Target episode missing from native action trace: " + repr((cohort, key)))
        if key not in source["calls_by_episode"]:
            raise ValueError("Target episode missing from S2 capture: " + repr((cohort, key)))
        chosen, run_summary = select_target_calls(source["calls_by_episode"], spec)
        target_plans.append((spec, source, chosen, run_summary))
        for item in chosen:
            call = item["call"]
            for image_index, ref in enumerate(call["images"]):
                source_image, source_sha = verify_source_image(source["capture_root"], ref)
                prepared_images.append({
                    "spec": spec, "source": source, "call": call, "item": item,
                    "image_index": image_index, "ref": ref,
                    "source_image": source_image, "source_file_sha256": source_sha,
                })

    output.mkdir(parents=True, exist_ok=False)
    image_rows = []
    selected_rows = []
    for spec, source, chosen, run_summary in target_plans:
        scene, episode = spec["key"]
        for item in chosen:
            call = item["call"]
            exported = []
            call_images = [
                x for x in prepared_images
                if x["source"]["cohort"] == spec["cohort"]
                and tuple(x["call"]["episode_key"]) == tuple(spec["key"])
                and int(x["call"]["call_id"]) == int(call["call_id"])
            ]
            for prepared in call_images:
                ref = prepared["ref"]
                image_index = prepared["image_index"]
                digest = ref["pixel_sha256"]
                folder = Path("images") / spec["cohort"] / (scene + "_" + episode)
                filename = "call_{:04d}_img_{:02d}_{}.png".format(
                    int(call["call_id"]), image_index, digest[:12])
                relative_output = folder / filename
                destination = output / relative_output
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(prepared["source_image"]), str(destination))

                if call["call_type"] == "ordinary":
                    role = "ordinary_current_rgb" if image_index == len(call["images"]) - 1 else "ordinary_history_context"
                else:
                    role = "lookdown_followup_last_input" if image_index == len(call["images"]) - 1 else "lookdown_followup_context"
                row = {
                    "cohort": spec["cohort"], "scene_id": scene, "episode_id": episode,
                    "target_role": spec["role"], "call_id": int(call["call_id"]),
                    "call_type": call["call_type"], "main_step": int(call["main_step"]),
                    "action_prefix_count_global": int(call["action_prefix_count_global"]),
                    "action_prefix_count_episode": int(call["action_prefix_count_episode"]),
                    "model_output": call["model_output"],
                    "native_parsed_actions": call["native_parsed_actions"],
                    "image_index_in_model_input": image_index,
                    "image_role": role, "pixel_sha256": digest,
                    "source_relative_path": ref["path"],
                    "source_image_path": str(prepared["source_image"]),
                    "source_file_sha256": prepared["source_file_sha256"],
                    "export_relative_path": relative_output.as_posix(),
                    "selection_reasons": item["reasons"],
                }
                image_rows.append(row)
                exported.append({
                    "image_index": image_index,
                    "pixel_sha256": digest,
                    "image_role": role,
                    "export_relative_path": relative_output.as_posix(),
                })
            selected_record = dict(call)
            selected_record.update({
                "cohort": spec["cohort"],
                "target_role": spec["role"],
                "action_prefix_count_global": int(call["action_prefix_count_global"]),
                "action_prefix_count_episode": int(call["action_prefix_count_episode"]),
                "selection_reasons": item["reasons"],
                "exported_images": exported,
            })
            selected_rows.append(selected_record)

    jsonl_write(output / "selected_calls.jsonl", selected_rows)
    jsonl_write(output / "image_manifest.jsonl", image_rows)

    target_manifest = []
    for spec, source, chosen, run_summary in target_plans:
        target_manifest.append({
            "cohort": spec["cohort"], "episode_key": list(spec["key"]),
            "role": spec["role"], "evidence_window_episode_actions": list(spec["window"]),
            "event_start_action": spec.get("event_start"),
            "event_end_action": spec.get("event_end"),
            "primary_pattern": spec["primary"],
            "secondary_pattern": spec.get("secondary"),
            "run_summary": run_summary,
            "selected_call_ids": [int(x["call"]["call_id"]) for x in chosen],
            "native_action_count": source["action_lengths"][tuple(spec["key"])],
            "native_action_global_start": source["action_offsets"][tuple(spec["key"])],
        })
    export_manifest = {
        "schema": SCHEMA,
        "purpose": "Post-hoc visual evidence inspection only; not training labels or online selector inputs",
        "cpu_only": True, "gpu_rollouts_executed": False,
        "original_data_modified": False, "source_files_modified": False,
        "image_verification": "PIL pixel digest recomputed exactly as recovery_task_evidence_acquire.pixel_digest",
        "action_offset_semantics": "native_action_prefix_count is run-global; localized by exact contiguous episode blocks in control/actions.jsonl",
        "cohorts": {
            name: {
                "paired_protocol": str(src["protocol_path"]),
                "native_run": str(src["native_root"]),
                "actions_sha256": src["actions_sha256"],
                "online_decisions_sha256": src["decisions_sha256"],
                "native_action_rows": src["action_total"],
            } for name, src in cohort_cache.items()
        },
        "targets": target_manifest,
        "N_selected_calls": len(selected_rows),
        "N_exported_image_occurrences": len(image_rows),
        "N_unique_pixel_hashes": len(set(x["pixel_sha256"] for x in image_rows)),
        "files": ["selected_calls.jsonl", "image_manifest.jsonl", "images/"],
    }
    (output / "export_manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    zip_path = output / "selected_rgb_s2_evidence.zip"
    with zipfile.ZipFile(str(zip_path), "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(str(path), arcname=path.relative_to(output).as_posix())

    print("NATIVE_S2_VISUAL_EVIDENCE_EXPORT=PASS")
    print("TARGET_EPISODES:", len(target_plans))
    print("SELECTED_CALLS:", len(selected_rows))
    print("IMAGE_OCCURRENCES:", len(image_rows))
    print("UNIQUE_PIXEL_HASHES:", export_manifest["N_unique_pixel_hashes"])
    for target in target_manifest:
        print("{} {} primary_run={} secondary_run={} selected_calls={}".format(
            target["cohort"], "/".join(target["episode_key"]),
            target["run_summary"]["primary_run_length"],
            target["run_summary"]["secondary_run_length"],
            target["run_summary"]["selected_call_count"]))
    print("OUTPUT_DIR:", output)
    print("UPLOAD_ZIP:", zip_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
