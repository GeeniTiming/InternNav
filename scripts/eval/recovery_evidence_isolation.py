"""Mechanism pilot: what does a physical turn change in frozen DualVLN?

Uses already *completed* independent checkpoints and their prior paired factorial:
  A RELEASE                           (reused)
  B LEFT45_KEEP / RIGHT45_KEEP        (reused, turn and remain facing outward)
  C LOOKBACK_{left,right}45_EVIDENCE  (three turns out, three back, add actual
                                      outward RGB to exactly one System-2 prompt)
  D LOOKBACK_{left,right}45_PLACEBO   (identical physical scan, but add duplicate
                                      current RGB under the exact same prompt)
  E HISTORY_RESET                     (reused)

C/D isolate the *content* of the supplementary image from the physical turn
sequence, pose, prompt wording, image count and charged action budget.
They do NOT prove task relevance, semantic grounding, or causal mediation.

Do not alter original native evaluator, checkpoint, or factorial source.
Source manifest and checkpoints must pass strict integrity validation.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import functools
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_independent_factorial as prior

pilot = prior.pilot
physical = prior.physical
checkpoint = prior.checkpoint
capture = prior.capture
common = prior.common
pose = prior.pose
turn = prior.turn

ARMS = (
    "lookback_left45_evidence",
    "lookback_left45_placebo",
    "lookback_right45_evidence",
    "lookback_right45_placebo",
)
ARM_ANGLES = {
    "lookback_left45_evidence": 45,
    "lookback_left45_placebo": 45,
    "lookback_right45_evidence": -45,
    "lookback_right45_placebo": -45,
}
SCAN_ACTIONS = 6  # native 3*15 degree yaw away, then 3*15 degrees back
POSE_TOLERANCE_M = 1e-5
HEADING_TOLERANCE_DEG = .05
EXTRA_PROMPT = " Supplementary visual input: <image>."
SCOPE = (
    "Two-scene engineering/mechanism pilot only. A/KEEP/RESET reused, symmetric "
    "45-degree outbound-and-return physical action pairs, extra real rotated "
    "view vs duplicated current-frame placebo with same prompt/image budget. "
    "No trained selector, automatic task-relevance judgment or held-out efficacy."
)


def arm_pair(arm):
    if arm not in ARMS:
        raise ValueError("Unknown mechanism arm " + str(arm))
    return ("left45" if ARM_ANGLES[arm] > 0 else "right45")


def validate_prior(checkpoint_run, factorial_run):
    """Validate only completed frozen source runs; never rewrite earlier reports."""
    old, cases = turn.source_check(checkpoint_run)
    _, linked_source, protocol = prior.read_run(factorial_run)
    if linked_source.resolve() != checkpoint_run.resolve():
        raise ValueError("The prior factorial used a different checkpoint source")
    if protocol["cases"] != cases["cases"]:
        raise ValueError("Prior factorial and captured checkpoints disagree")
    report = pilot.read_json(factorial_run / "candidate_report.json")
    if (report.get("schema") != "recovery_independent_factorial_report.v1"
            or report.get("status") != "READY_FOR_INTERPRETATION"):
        raise ValueError("The previous 5x2 factorial report is not complete")
    expected = len(cases["cases"]) * len(prior.ARMS)
    rows = report.get("outcomes", [])
    if len(rows) != expected:
        raise ValueError("Prior factorial lacks expected candidate labels")
    for required in ("history_reset", "left45_keep", "right45_keep"):
        if pilot.read_json(factorial_run / required / "pilot_summary.json")["status"] != "complete":
            raise ValueError("Prior comparator arm incomplete: " + required)
    return old, cases, report


def prepare(args):
    source = Path(args.source_run).resolve()
    factorial = Path(args.factorial_run).resolve()
    root = Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError("Create-only mechanism pilot; use a fresh run-dir")
    old, cases, _ = validate_prior(source, factorial)
    cfg = pilot.load_config(source / "control.py")
    if (cfg.agent.model_settings.get("mode") != "dual_system"
            or cfg.agent.model_settings.get("kv_cache_continuation", False)):
        raise ValueError("Frozen original DualVLN with cache disabled is required")
    limit = int(cfg.eval_settings["max_steps_per_episode"])
    if any(int(c["checkpoint"]["step"]) + SCAN_ACTIONS >= limit for c in cases["cases"]):
        raise ValueError("Not enough remaining main-loop budget for a yaw scan")
    protocol = {
        "schema": "recovery_evidence_isolation.v1",
        "source_run": str(source),
        "factorial_run": str(factorial),
        "cases": cases["cases"],
        "arms": list(ARMS),
        "angles": ARM_ANGLES,
        "forced_native_actions": SCAN_ACTIONS,
        "original_main_loop_limit": limit,
        "charged_main_loop_limit": limit - SCAN_ACTIONS,
        "injection": {
            "location": "first ordinary System-2 prompt immediately after scan",
            "image_position": "extra image immediately before current image",
            "prompt": EXTRA_PROMPT,
            "evidence": "outward RGB at 45 degrees resized to model dimensions",
            "placebo": "the current model-sized RGB repeated as extra image",
            "same_number_of_images": True,
            "no_changes_to_persistent_rgb_list": True,
            "no_changes_to_instruction_or_model_weights": True,
        },
        "limitations": SCOPE,
    }
    revision = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    manifest = {
        "schema": protocol["schema"],
        "seed_base": old["seed_base"],
        "episodes": old["episodes"],
        "revision": revision,
        "source_hashes": dict(old["source_hashes"]),
        "generated_hashes": {},
        "checkpoint_shard_sizes": old["checkpoint_shard_sizes"],
        "evidence_isolation": {"source_run": str(source), "factorial_run": str(factorial)},
    }
    manifest["source_hashes"].update(old["generated_hashes"])
    watched = [
        source / "manifest.json", source / "checkpoint_report.json",
        factorial / "manifest.json", factorial / "candidate_report.json",
        Path(__file__).resolve(), Path(prior.__file__).resolve(),
        Path(turn.__file__).resolve(), Path(checkpoint.__file__).resolve(),
        Path(physical.__file__).resolve(), Path(pose.__file__).resolve(),
        Path(common.__file__).resolve(),
    ]
    watched.extend(
        p for p in (source / "control").rglob("*")
        if p.is_file() and p.suffix in (".json", ".jsonl", ".png")
    )
    for arm in ("history_reset", "left45_keep", "right45_keep"):
        watched.extend([
            factorial / arm / "pilot_summary.json",
            factorial / arm / "turn_summary.json",
            factorial / arm / "progress.json",
        ])
    manifest["source_hashes"].update({str(p): pilot.sha256(p) for p in watched})
    root.mkdir(parents=True, exist_ok=False)
    p = root / "mechanism_protocol.json"
    pilot.write_json(p, protocol)
    manifest["generated_hashes"][str(p)] = pilot.sha256(p)
    for arm in ARMS:
        c = copy.deepcopy(cfg)
        c.eval_settings["output_path"] = str(root / arm)
        c.agent.model_settings["vis_debug"] = True
        c.agent.model_settings["vis_debug_path"] = str(root / "video" / arm)
        path = root / (arm + ".py")
        value = json.dumps(c.model_dump(mode="json", exclude_none=True))
        path.write_text(
            "import json\nfrom internnav.configs.evaluator import EvalCfg\n"
            "eval_cfg = EvalCfg.model_validate(json.loads(" + repr(value) + "))\n",
            encoding="utf-8",
        )
        manifest["generated_hashes"][str(path)] = pilot.sha256(path)
    pilot.write_json(root / "manifest.json", manifest)
    print("EVIDENCE_ISOLATION_PREPARE=PASS", len(cases["cases"]), "contexts; new arms=", len(ARMS))
    print("Source and original 5x2 results reused; no GPU/model loaded")


def read_run(root):
    manifest = pilot.verify_manifest(root)
    links = manifest["evidence_isolation"]
    source = Path(links["source_run"])
    factorial = Path(links["factorial_run"])
    _, cases, _ = validate_prior(source, factorial)
    protocol = pilot.read_json(root / "mechanism_protocol.json")
    if (protocol["schema"] != "recovery_evidence_isolation.v1"
            or protocol["cases"] != cases["cases"]
            or protocol["arms"] != list(ARMS)
            or protocol["angles"] != ARM_ANGLES
            or protocol["forced_native_actions"] != SCAN_ACTIONS
            or protocol["injection"]["prompt"] != EXTRA_PROMPT):
        raise ValueError("Locked evidence-isolation protocol changed")
    return source, factorial, protocol


def instrument_loop(function):
    """Runtime-only guarded AST hooks: native evaluator remains untouched."""
    if function.__code__.co_freevars:
        raise ValueError("Unexpected evaluator closure")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    loops = [
        node for node in ast.walk(tree) if isinstance(node, ast.While)
        and any(isinstance(n, ast.Attribute) and n.attr == "max_steps_per_episode"
                for n in ast.walk(node.test))
    ]
    if len(loops) != 1:
        raise ValueError("Expected exactly one native episode step loop")
    loop = loops[0]
    before = ast.parse(
        "observations, done = self._recovery_candidate.on_boundary("
        "step_id, observations, done, rgb_list, action_seq, pixel_goal, local_actions, action)"
    ).body[0]
    loop.body.insert(0, before)
    history = ast.parse(
        "history_id = np.unique(np.linspace(0, step_id - 1, "
        "self.num_history, dtype=np.int32)).tolist()"
    ).body[0]
    images = ast.parse(
        "input_images = [rgb_list[i] for i in history_id] + cur_images"
    ).body[0]
    response = ast.parse(
        "llm_outputs = self.processor.tokenizer.decode("
        "output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)"
    ).body[0]
    h_found, i_found, r_found = 0, 0, 0
    # Locate assignments structurally; fail closed if evaluator source changes.
    for node in ast.walk(loop):
        for field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, item in reversed(list(enumerate(value))):
                if isinstance(item, ast.Assign) and ast.dump(item) == ast.dump(history):
                    value.insert(index + 1, ast.parse(
                        "history_id = self._recovery_candidate.filter_history_ids("
                        "step_id, history_id, len(rgb_list))"
                    ).body[0])
                    h_found += 1
                if isinstance(item, ast.Assign) and ast.dump(item) == ast.dump(images):
                    value.insert(index + 1, ast.parse(
                        "self._recovery_candidate.inject_extra_view("
                        "step_id, sources, input_images)"
                    ).body[0])
                    i_found += 1
                if isinstance(item, ast.Assign) and ast.dump(item) == ast.dump(response):
                    value.insert(index + 1, ast.parse(
                        "self._recovery_candidate.record_system2_output(step_id, llm_outputs)"
                    ).body[0])
                    r_found += 1
    if (h_found, i_found, r_found) != (1, 1, 1):
        raise RuntimeError(
            "NOT_COMPARABLE: expected one native history/image/response assembly, found "
            + repr((h_found, i_found, r_found))
        )
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, "<evidence-isolation-native-loop>", "exec"), namespace)
    return functools.update_wrapper(namespace[function.__name__], function)


class MechanismProbe(prior.CandidateProbe):
    def begin(self, key):
        super().begin(key)
        self.inject_pending = False
        self.inject_count = 0
        self.inject_events = []
        self.view_image = None
        self.view_sha256 = None
        self.return_sha256 = None
        self.first_system2_response = None
        self.reset_enabled = False

    def on_boundary(self, step, observations, done, history, action_seq, pixel_goal, local_actions, action):
        ep = self.evaluator.env.get_current_episode()
        key = capture.episode_key({"scene_id": ep.scene_id, "episode_id": ep.episode_id})
        if key != self.current:
            self.finish()
            self.begin(key)
        self.current_step = int(step)
        expected = self.cases[key]["checkpoint"]
        if self.gate is not None or step < expected["step"]:
            return observations, done
        if (int(step) != expected["step"] or len(self.records) != expected["prefix_action_count"]
                or done or action_seq or pixel_goal is not None
                or local_actions or action not in (0, 1, 2, 3)):
            raise RuntimeError("NOT_COMPARABLE: locked idle System-2 boundary changed")
        pitch = checkpoint.camera_pitch_degrees(checkpoint.quaternion_list(
            self.evaluator.env._env.sim.get_agent_state().sensor_states["rgb"].rotation
        ))
        if abs(pitch) > 1e-3:
            raise RuntimeError("NOT_COMPARABLE: RGB camera not level")
        gate = {
            "step": int(step), "prefix_action_count": len(self.records),
            "state": self.state(),
            "observation_hashes": physical.observation_hashes(observations),
            "history_length": len(history),
            "history_sha256": physical.history_hash(history),
            "rng_sha256": common.rng_digest(),
            "parameters_sha256": self.parameters,
            "prefix_trace_sha256": checkpoint.cohort.digest(self.records),
        }
        common.same_fields(gate, expected, tuple(gate), "mechanism boundary")
        if common.parameter_versions(self.evaluator.model) != self.parameters:
            raise RuntimeError("Model parameter versions changed before scan")
        self.gate, self.history_prefix = gate, tuple(history)

        latest = [observations]
        self.phase = "external_recovery"
        origin = gate["state"]["orientation_wxyz"]
        outbound_angle = ARM_ANGLES[self.arm]
        outward_target = pose.yaw_quaternion(pose.yaw_degrees(origin) + outbound_angle)

        def advance(act):
            result = self.evaluator.env.step(act)
            latest[0] = result[0]
            return result

        from PIL import Image
        with physical.preserve_policy_rng():
            outbound = pose.turn_to_heading(outward_target, self.state, advance, budget=3)
            if outbound != 3:
                raise RuntimeError("NOT_COMPARABLE: outbound scan required !=3 turns")
            outward_state = self.state()
            if pose.heading_error(outward_state["orientation_wxyz"], outward_target) > HEADING_TOLERANCE_DEG:
                raise RuntimeError("Outbound image pose differs from declared view")
            self.view_image = Image.fromarray(latest[0]["rgb"]).convert("RGB").resize(
                (self.evaluator.model_args.resize_w, self.evaluator.model_args.resize_h)
            )
            self.view_sha256 = physical.history_hash([self.view_image])
            inbound = pose.turn_to_heading(origin, self.state, advance, budget=3)
            if inbound != 3:
                raise RuntimeError("NOT_COMPARABLE: inbound scan required !=3 turns")

        after_state = self.state()
        self.external_actions = len(self.records) - expected["prefix_action_count"]
        if (self.external_actions != SCAN_ACTIONS
                or math.dist(after_state["position"], gate["state"]["position"]) > POSE_TOLERANCE_M
                or pose.heading_error(after_state["orientation_wxyz"], origin) > HEADING_TOLERANCE_DEG):
            raise RuntimeError("NOT_COMPARABLE: scan did not restore the physical pose")
        if (common.rng_digest() != gate["rng_sha256"]
                or physical.history_hash(history) != gate["history_sha256"]
                or common.parameter_versions(self.evaluator.model) != self.parameters):
            raise RuntimeError("NOT_COMPARABLE: scan changed RNG, original history, or weights")
        self.evaluator.max_steps_per_episode = self.original_limit - SCAN_ACTIONS
        if self.evaluator.max_steps_per_episode <= step:
            raise RuntimeError("Insufficient charged continuation budget")
        self.phase = "continuation"
        self.inject_pending = True
        self.handoff = {
            "state": after_state, "external_actions": self.external_actions,
            "outward_angle_deg": outbound_angle,
            "outward_view_pixel_sha256": self.view_sha256,
            "history_reset": False, "history_cutoff": None,
            "policy_rng_restored": True, "model_unchanged": True,
            "original_history_unchanged": True,
            "eligible": True, "charged_native_limit": self.evaluator.max_steps_per_episode,
            "final_observation_hashes": physical.observation_hashes(latest[0]),
        }
        folder = self.directory / "checkpoints" / (key[0] + "_" + key[1])
        folder.mkdir(parents=True, exist_ok=False)
        pilot.write_json(folder / "boundary.json", gate)
        pilot.write_json(folder / "handoff.json", self.handoff)
        self.view_image.save(folder / "outward_view_model_rgb.png")
        Image.fromarray(latest[0]["rgb"]).save(folder / "after_return.png")
        print("SCAN_AND_RETURN=PASS", self.arm, key, "actions=", self.external_actions)
        return latest[0], False

    def inject_extra_view(self, step, sources, input_images):
        if not self.inject_pending:
            return
        if (self.phase != "continuation" or self.inject_count != 0
                or len(input_images) < 1 or int(step) != int(self.gate["step"])):
            raise RuntimeError("NOT_COMPARABLE: supplementary view injected at wrong decision")
        image = (self.view_image if self.arm.endswith("_evidence")
                 else input_images[-1].copy())
        if image.size != input_images[-1].size or image.mode != input_images[-1].mode:
            raise RuntimeError("Supplementary view and placebo dimensions differ")
        sources[0]["value"] += EXTRA_PROMPT
        input_images.insert(-1, image)
        # The native evaluator appends exactly one further <image> for the
        # current view immediately after this hook. Fail on token/image drift.
        if sources[0]["value"].count("<image>") + 1 != len(input_images):
            raise RuntimeError("NOT_COMPARABLE: supplementary token/image count mismatch")
        self.inject_count += 1
        self.inject_pending = False
        self.inject_events.append({
            "key": list(self.current), "main_step": int(step),
            "source": "outward_view" if self.arm.endswith("_evidence") else "duplicate_current",
            "supplementary_pixel_sha256": physical.history_hash([image]),
            "native_history_unchanged": True,
            "same_image_count": True,
            "insertion_order": "before_original_current",
            "prompt": EXTRA_PROMPT,
        })
        self.return_sha256 = physical.history_hash([input_images[-1]])
        print("SUPPLEMENTARY_VIEW_INJECTED=PASS", self.arm, self.current, step)

    def record_system2_output(self, step, message):
        if self.inject_count == 1 and self.first_system2_response is None:
            self.first_system2_response = {
                "key": list(self.current), "step": int(step), "text": str(message),
            }

    def finish(self):
        key = self.current
        if key is None:
            return
        if self.inject_pending or self.inject_count != 1 or self.first_system2_response is None:
            raise RuntimeError("NOT_COMPARABLE: exactly one first-System-2 injection required")
        super().finish()
        data = self.finished[key]
        data["injection_events"] = self.inject_events
        data["first_system2_response"] = self.first_system2_response
        folder = self.directory / "checkpoints" / (key[0] + "_" + key[1])
        pilot.write_json(folder / "evidence_input.json", {
            "schema": "evidence_input.v1",
            "events": self.inject_events, "first_system2_response": self.first_system2_response,
            "scan_view_sha256": self.view_sha256, "post_return_current_sha256": self.return_sha256,
        })


def run(args):
    root = Path(args.run_dir).resolve()
    source, _, protocol = read_run(root)
    if args.arm not in ARMS:
        raise ValueError("Unknown arm")
    if (root / args.arm).exists():
        raise FileExistsError("Create-only arm already exists")
    references = {tuple(c["key"]): [] for c in protocol["cases"]}
    for row in physical.read_lines(source / "control/actions.jsonl"):
        references[capture.episode_key(row)].append(row)

    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators["habitat_vln"]
    original_init, original_loop = cls.__init__, cls._run_eval_dual_system
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get("kv_cache_continuation", False):
            raise RuntimeError("Require frozen native cache-disabled model")
        original_init(evaluator, cfg)
        probe = MechanismProbe(evaluator, root, args.arm, protocol, references)
        probes.append(probe)
        evaluator._recovery_candidate = probe
        evaluator.env.step = probe.record
        probe.original_generate = evaluator.model.generate

        def counted_generate(*a, **kw):
            probe.s2_calls += 1
            return probe.original_generate(*a, **kw)

        evaluator.model.generate = counted_generate

    cls.__init__, cls._run_eval_dual_system = initialize, instrument_loop(original_loop)
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError("Expected exactly one native evaluator")
        probes[0].finish()
        if set(probes[0].finished) != set(references):
            raise RuntimeError("NOT_COMPARABLE: not all predeclared scenes finished")
        probes[0].status = "complete"
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = original_init, original_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.evaluator.model.generate = probe.original_generate
            probe.close(error)


def report(args):
    root = Path(args.run_dir).resolve()
    source, old_factorial, protocol = read_run(root)
    prior_report = pilot.read_json(old_factorial / "candidate_report.json")
    rows = {tuple(c["key"]): {"key": c["key"]} for c in protocol["cases"]}
    from_prior = {
        (tuple(row["key"]), row["arm"]): row
        for row in prior_report["outcomes"]
    }
    baseline = capture.read_progress(source / "control/progress.json")

    for key in rows:
        rows[key]["release"] = int(baseline[key]["success"])
        for old in ("history_reset", "left45_keep", "right45_keep"):
            rows[key][old] = int(from_prior[(key, old)]["success"])

    ref_traces = {tuple(c["key"]): [] for c in protocol["cases"]}
    for r in physical.read_lines(source / "control/actions.jsonl"):
        ref_traces[capture.episode_key(r)].append(r)

    scan_traces = {}
    for arm in ARMS:
        summary = pilot.read_json(root / arm / "pilot_summary.json")
        check = pilot.read_json(root / arm / "turn_summary.json")
        if summary["status"] != "complete" or check["status"] != "complete":
            raise RuntimeError("Incomplete mechanism arm " + arm)
        native_summary = pilot.read_json(source / "control/pilot_summary.json")
        common.same_fields(summary, native_summary, common.RUNTIME_FIELDS, "mechanism runtime")
        progress = capture.read_progress(root / arm / "progress.json")
        details = {tuple(c["key"]): c for c in check["cases"]}
        tagged = physical.validate_trace(
            physical.read_lines(root / arm / "turn_actions.jsonl"), arm
        )
        native = physical.validate_trace(
            physical.read_lines(root / arm / "actions.jsonl"), arm + " native"
        )
        if not pilot.compare_traces(tagged, native)["actions_and_poses_match"]:
            raise RuntimeError("Native/observer trace mismatch in " + arm)
        if set(progress) != set(rows) or set(details) != set(rows):
            raise RuntimeError("Mechanism arm changed the locked cohort")
        for key in rows:
            detail = details[key]
            checkpoint_data = next(c["checkpoint"] for c in protocol["cases"]
                                   if tuple(c["key"]) == key)
            n = checkpoint_data["prefix_action_count"]
            trace = [t for t in tagged if capture.episode_key(t) == key]
            same = pilot.compare_traces(ref_traces[key][:n], trace[:n])
            if not same["actions_and_poses_match"]:
                raise RuntimeError("Native prefix differs in " + arm + " " + str(key))
            external = [r for r in trace if r["phase"] == "external_recovery"]
            if (len(external) != SCAN_ACTIONS or
                    [r["action"] for r in external] !=
                    ([2, 2, 2, 3, 3, 3] if ARM_ANGLES[arm] > 0 else [3, 3, 3, 2, 2, 2])):
                raise RuntimeError("Not exactly six declared native scan actions")
            if any(math.dist(r["position"], checkpoint_data["state"]["position"])
                   > POSE_TOLERANCE_M for r in external):
                raise RuntimeError("Scan translated instead of rotating in place")
            if pose.heading_error(external[-1]["orientation_wxyz"],
                                  checkpoint_data["state"]["orientation_wxyz"]) > HEADING_TOLERANCE_DEG:
                raise RuntimeError("Scan did not return to the original heading")
            records = detail["injection_events"]
            evidence = pilot.read_json(
                root / arm / "checkpoints" / (key[0] + "_" + key[1])
                / "evidence_input.json"
            )
            if (len(records) != 1 or records != evidence["events"] or
                    records[0]["prompt"] != EXTRA_PROMPT or
                    records[0]["main_step"] != checkpoint_data["step"]):
                raise RuntimeError("Supplementary input not injected exactly once")
            folder = root / arm / "checkpoints" / (key[0] + "_" + key[1])
            from PIL import Image
            with Image.open(folder / "outward_view_model_rgb.png") as opened:
                recorded_view_hash = physical.history_hash([opened.copy()])
            if recorded_view_hash != evidence["scan_view_sha256"]:
                raise RuntimeError("Outward RGB hash differs from captured view")
            expected_source = (
                "outward_view" if arm.endswith("_evidence") else "duplicate_current"
            )
            if records[0]["source"] != expected_source:
                raise RuntimeError("Wrong visual input source in mechanism arm")
            expected_pixel_hash = (evidence["scan_view_sha256"] if
                                   arm.endswith("_evidence") else
                                   evidence["post_return_current_sha256"])
            if records[0]["supplementary_pixel_sha256"] != expected_pixel_hash:
                raise RuntimeError("Supplementary image provenance/pixels changed")
            scan_traces[(arm, key)] = external
            rows[key][arm] = {
                "success": int(progress[key]["success"]),
                "ne": progress[key]["ne"],
                "external_actions": len(external),
                "supplementary_view_source": records[0]["source"],
                "supplementary_sha256": records[0]["supplementary_pixel_sha256"],
                "post_return_current_sha256": evidence["post_return_current_sha256"],
                "first_system2_response": detail["first_system2_response"],
            }

    for side in ("left45", "right45"):
        e, p = ("lookback_" + side + "_evidence",
                "lookback_" + side + "_placebo")
        for key in rows:
            if not pilot.compare_traces(
                scan_traces[(e, key)], scan_traces[(p, key)]
            )["actions_and_poses_match"]:
                raise RuntimeError("C/D have different native physical scans")
            if rows[key][e]["post_return_current_sha256"] != rows[key][p]["post_return_current_sha256"]:
                raise RuntimeError("C/D current RGB differs after identical return")
            rows[key][side + "_novel_pixel_content"] = (
                rows[key][e]["supplementary_sha256"] != rows[key][p]["supplementary_sha256"]
            )
            rows[key][side + "_evidence_minus_placebo"] = (
                rows[key][e]["success"] - rows[key][p]["success"]
            )
    report = {
        "schema": "evidence_isolation_report.v1",
        "status": "MECHANISM_PILOT_VALIDATED",
        "N_contexts": len(rows),
        "new_rollouts": len(ARMS) * len(rows),
        "reused_controls": "RELEASE, HISTORY_RESET and left/right45_KEEP",
        "per_context": list(rows.values()),
        "scope": SCOPE,
        "interpretation_warning": (
            "C-D estimates extra image *content* under identical scan, prompt "
            "and action budget; it does not establish task relevance or exclude "
            "all downstream semantic/temporal confounders. B-C is not a pure heading effect."
        ),
    }
    pilot.write_json(root / "mechanism_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--source-run", required=True)
    prep.add_argument("--factorial-run", required=True)
    prep.add_argument("--run-dir", required=True)
    prep.set_defaults(function=prepare)
    for name, fn in (("run", run), ("run-all", None), ("report", report)):
        p = subs.add_parser(name)
        p.add_argument("--run-dir", required=True)
        if name == "run":
            p.add_argument("--arm", choices=ARMS, required=True)
        if name == "run-all":
            p.set_defaults(function=run_all)
        else:
            p.set_defaults(function=fn)
    args = parser.parse_args(argv)
    args.function(args)


def run_all(args):
    root = Path(args.run_dir).resolve()
    read_run(root)
    if any((root / arm).exists() for arm in ARMS):
        raise FileExistsError("run-all is create-only; run individual remaining arms if partial")
    for arm in ARMS:
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).resolve()), "run",
            "--run-dir", str(root), "--arm", arm
        ], check=True)
    report(args)


if __name__ == "__main__":
    main()
