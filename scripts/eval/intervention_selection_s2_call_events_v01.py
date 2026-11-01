"""Passive, append-only System-2 generation events for NEW paired replay arms.

This is a probe adapter, not a way to recover missing past S2 calls.
The owner must already record every real env action and verify the paired gate.
No GT position, goal distance, future route, or oracle labels are emitted.

Integration in a new per-arm replay:
    rec = ArmS2Events(probe, arm_dir / 's2_call_events.jsonl')
    original_generate = evaluator.model.generate
    evaluator.model.generate = rec.wrap(original_generate)
    try: ... run original replay ...
    finally: evaluator.model.generate = original_generate; rec.close()

The capture creates its file exclusively; never attach to historical arms.
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path, PurePosixPath
from collections import defaultdict

SCHEMA = "intervention_selection_s2_call_events.v0.1"
VALID_PHASES = frozenset(("prefix", "external_recovery", "alignment", "continuation"))


def _digest_tensor(tensor):
    """Small token tensor fingerprint, without mutating RNG or model inputs."""
    if tensor is None:
        return None
    if not hasattr(tensor, "detach"):
        raise TypeError("Expected model input_ids tensor")
    value = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(value.shape)).encode("ascii"))
    h.update(b"\0")
    h.update(str(value.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _record_jsonl(file, row):
    file.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    file.flush()


class ArmS2Events:
    """Exact model.generate entry/exit and same-arm env action count.

    `probe` must expose `current` canonical (scene,episode) tuple, `records`
    (one row per actual env.step), `phase`, `current_step`, `arm`, and evaluator
    with a processor tokenizer. No inferred call types or step-to-S2 mapping.
    """

    def __init__(self, probe, path):
        self.probe = probe
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8")
        self.next_call_id = 0
        self.episode_counts = defaultdict(int)
        self.pending = None
        self.closed = False

    def snapshot(self):
        key = self.probe.current
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise RuntimeError("S2 generate outside an active canonical episode")
        if not all(isinstance(x, str) and x for x in key):
            raise RuntimeError("S2 episode identity is not canonical")
        if len(self.probe.records) < 0:
            raise RuntimeError("Invalid action stream")
        phase = self.probe.phase
        if phase not in VALID_PHASES:
            raise RuntimeError("Unknown replay phase: " + repr(phase))
        if (not isinstance(self.probe.current_step, int)
                or isinstance(self.probe.current_step, bool)
                or self.probe.current_step < 0):
            raise RuntimeError("No actual loop main_step at generate entry")
        return {
            "episode_key": list(key), "arm": str(getattr(self.probe, "output_arm", self.probe.arm)),
            "phase": phase, "main_step": self.probe.current_step,
            "native_action_prefix_count": len(self.probe.records),
        }

    def wrap(self, original_generate):
        @functools.wraps(original_generate)
        def logged(*args, **kwargs):
            if self.closed or self.pending is not None:
                raise RuntimeError("S2 logger closed or nested generate call")
            anchor = self.snapshot()
            inp = kwargs.get("input_ids")
            if inp is None:
                raise RuntimeError("Cannot anchor native S2 without input_ids")
            cid = self.next_call_id
            key = tuple(anchor["episode_key"])
            episode_call_id = self.episode_counts[key]
            self.next_call_id += 1
            self.episode_counts[key] += 1
            before = {
                "schema": SCHEMA, "event": "enter", "call_id": cid,
                "episode_call_id": episode_call_id, **anchor,
                "input_ids_sha256": _digest_tensor(inp),
                "input_ids_shape": list(inp.shape),
                "model_input_keys": sorted(kwargs.keys()),
                "call_type": "NOT_OBSERVABLE_WITH_GENERATE_WRAPPER",
            }
            self.pending = before
            _record_jsonl(self.stream, before)
            try:
                result = original_generate(*args, **kwargs)
                sequences = result.sequences
                if len(sequences) != 1 or len(inp) != 1:
                    raise RuntimeError("Not a single-sequence native S2 generation")
                continuation = sequences[0][int(inp.shape[1]):]
                output = self.probe.evaluator.processor.tokenizer.decode(
                    continuation, skip_special_tokens=True)
                after_anchor = self.snapshot()
                if after_anchor != anchor:
                    raise RuntimeError("Replay state changed inside model.generate")
                finish = {
                    "schema": SCHEMA, "event": "exit", "call_id": cid,
                    "episode_call_id": episode_call_id, **after_anchor,
                    "model_output": str(output),
                    "model_output_sha256": hashlib.sha256(
                        str(output).encode("utf-8")).hexdigest(),
                    "generated_token_count": len(continuation),
                }
                _record_jsonl(self.stream, finish)
                self.pending = None
                return result
            except BaseException as exc:
                _record_jsonl(self.stream, {
                    "schema": SCHEMA, "event": "error", "call_id": cid,
                    "episode_call_id": episode_call_id, **anchor,
                    "error_type": type(exc).__name__,
                })
                self.pending = None
                raise
        return logged

    def close(self):
        if not self.closed:
            self.stream.close()
            self.closed = True


def audit_events(event_file, action_file, *, expected_arm=None):
    """CPU fail-closed structure + action-offset binding; not a paired gate audit.

    `action_file` must be the same-arm tagged env-step stream (not a different
    source-run KEEP). Does not turn these checks into causal identification.
    """
    with Path(action_file).open(encoding="utf-8") as f:
        actions = [json.loads(line) for line in f if line.strip()]
    action_counts = defaultdict(int)
    for row in actions:
        key = (PurePosixPath(str(row["scene_id"]).replace("\\", "/")).stem,
               str(row["episode_id"]))
        action_counts[key] += 1
    records = []
    with Path(event_file).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records or len(records) % 2:
        raise ValueError("Empty or partial S2 call events")
    last_offset = defaultdict(lambda: -1)
    count = 0
    for i in range(0, len(records), 2):
        enter, exit = records[i:i+2]
        if (enter.get("schema") != SCHEMA or exit.get("schema") != SCHEMA
                or enter.get("event") != "enter" or exit.get("event") != "exit"
                or enter.get("call_id") != count or exit.get("call_id") != count):
            raise ValueError("Missing/duplicate/failed S2 entry-exit pair")
        if any(enter.get(k) != exit.get(k) for k in (
            "episode_key", "arm", "phase", "main_step", "native_action_prefix_count",
            "episode_call_id")):
            raise ValueError("Call entry/exit changed anchor")
        key = tuple(enter["episode_key"])
        offset = enter["native_action_prefix_count"]
        if (not isinstance(offset, int) or isinstance(offset, bool) or offset < last_offset[key]
                or offset > action_counts[key] or enter["phase"] not in VALID_PHASES):
            raise ValueError("Invalid same-arm per-episode action offset")
        if expected_arm is not None and enter["arm"] != expected_arm:
            raise ValueError("Wrong S2 arm identity")
        if (not isinstance(exit.get("model_output"), str)
                or hashlib.sha256(exit["model_output"].encode("utf-8")).hexdigest()
                != exit.get("model_output_sha256")):
            raise ValueError("Invalid model output hash")
        last_offset[key] = offset
        count += 1
    return {"schema": SCHEMA, "status": "CALL_EVENT_AND_ACTION_OFFSETS_VALIDATED_GATE_NOT_YET_PAIRED",
            "N_calls": count, "N_action_rows": len(actions),
            "N_episodes": len(last_offset), "paired_gate_verified": False,
            "same_arm_action_sha256": hashlib.sha256(Path(action_file).read_bytes()).hexdigest(),
            "call_event_sha256": hashlib.sha256(Path(event_file).read_bytes()).hexdigest()}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit",))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    arm = root / args.arm
    events = arm / "s2_call_events.jsonl"
    actions = arm / "turn_actions.jsonl"
    output = arm / "s2_call_events_audit.json"
    if output.exists():
        parser.error("Audit report is create-only; do not overwrite previous evidence")
    result = audit_events(events, actions, expected_arm=args.arm)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    print("S2_CALL_EVENTS_V01=PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
