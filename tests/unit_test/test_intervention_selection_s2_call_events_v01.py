"""CPU-only S2 replay event contract tests; no model weights or GPU required."""
from __future__ import annotations
import json
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import intervention_selection_s2_call_events_v01 as s2

try:
    import torch
except ImportError:
    torch = None


class DummyTokens:
    def decode(self, toks, skip_special_tokens=True):
        return "LEFT" if len(toks) == 2 else "RIGHT"


def probe():
    return SimpleNamespace(
        current=("i5noydFURQK", "70"), arm="lookback_right45_evidence",
        output_arm="perturbed_seek_right45",
        records=[{"action": 1}], phase="continuation", current_step=24,
        evaluator=SimpleNamespace(processor=SimpleNamespace(tokenizer=DummyTokens())))


def run_call(rec, *, output_tokens=2):
    def generate(*args, **kwargs):
        # Fake pure function; no action steps or controller state changes.
        return SimpleNamespace(sequences=torch.tensor([[11,12] + [13]*output_tokens]))
    wrapped = rec.wrap(generate)
    return wrapped(input_ids=torch.tensor([[11,12]],dtype=torch.int64), pixel_values=None)


@unittest.skipIf(torch is None, "torch unavailable")
class CaptureTests(unittest.TestCase):
    def test_entry_exit_real_offsets_and_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"s2_calls.jsonl"
            p=probe()
            r=s2.ArmS2Events(p,path)
            self.assertEqual(run_call(r).sequences.shape[1],4)
            p.records.append({"action":3})
            p.current_step+=1
            self.assertEqual(run_call(r,output_tokens=1).sequences.shape[1],3)
            r.close()
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([x["event"] for x in rows],["enter","exit","enter","exit"])
            self.assertEqual([x["native_action_prefix_count"] for x in rows],[1,1,2,2])
            self.assertEqual([x["call_id"] for x in rows],[0,0,1,1])
            self.assertEqual(rows[0]["arm"],"perturbed_seek_right45")
            self.assertEqual(rows[1]["model_output"],"LEFT")
            self.assertNotIn("distance_to_goal",str(rows))
            actions=Path(tmp)/"turn_actions.jsonl"
            actions.write_text("".join(json.dumps({"scene_id":"dataset/i5noydFURQK.glb", "episode_id":"70", "action":x})+"\n" for x in (1,3)))
            result=s2.audit_events(path,actions,expected_arm="perturbed_seek_right45")
            self.assertEqual(result["N_calls"],2)
            self.assertFalse(result["paired_gate_verified"])

    def test_same_step_two_calls_can_have_distinct_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=probe()
            r=s2.ArmS2Events(p,Path(tmp)/"events")
            run_call(r);run_call(r);r.close()
            rows=[json.loads(x) for x in (Path(tmp)/"events").read_text().splitlines()]
            self.assertEqual([rows[0]["call_id"],rows[2]["call_id"]],[0,1])
            self.assertEqual(rows[0]["main_step"], rows[2]["main_step"])
            self.assertEqual(rows[0]["native_action_prefix_count"],rows[2]["native_action_prefix_count"])

    def test_incomplete_generation_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=probe();r=s2.ArmS2Events(p,Path(tmp)/"events")
            def broken(**kwargs):raise RuntimeError("model failed")
            with self.assertRaisesRegex(RuntimeError,"model failed"):
                r.wrap(broken)(input_ids=torch.tensor([[1]]))
            r.close()
            actions=Path(tmp)/"actions";actions.write_text('{"scene_id":"i5noydFURQK.glb","episode_id":"70"}\n')
            with self.assertRaisesRegex(ValueError,"partial|failed"):
                s2.audit_events(Path(tmp)/"events",actions)

    def test_fails_closed_missing_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=probe();p.current=None
            r=s2.ArmS2Events(p,Path(tmp)/"events")
            with self.assertRaisesRegex(RuntimeError,"active canonical"):
                run_call(r)
            r.close()

    def test_no_write_over_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"events";path.write_text("previous\n")
            with self.assertRaises(FileExistsError):s2.ArmS2Events(probe(),path)
            self.assertEqual(path.read_text(),"previous\n")

    def test_fail_on_modified_action_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=probe();e=Path(tmp)/"events"
            r=s2.ArmS2Events(p,e);run_call(r);r.close()
            actions=Path(tmp)/"actions"
            actions.write_text(json.dumps({"scene_id":"i5noydFURQK.glb","episode_id":"70","action":1})+"\n")
            rows=[json.loads(x) for x in e.read_text().splitlines()]
            rows[0]["native_action_prefix_count"]=6
            e.write_text("".join(json.dumps(x)+"\n" for x in rows))
            with self.assertRaises(ValueError):s2.audit_events(e,actions)

    def test_entry_exit_bind_to_source_exact_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=probe();e=Path(tmp)/"events";r=s2.ArmS2Events(p,e);run_call(r);r.close()
            actions=Path(tmp)/"actions"
            actions.write_text(json.dumps({"scene_id":"i5noydFURQK.glb","episode_id":"70","action":1})+"\n")
            rows=[json.loads(x) for x in e.read_text().splitlines()]
            rows[1]["model_output"]="tampered"
            e.write_text("".join(json.dumps(x)+"\n" for x in rows))
            with self.assertRaisesRegex(ValueError,"output hash"):s2.audit_events(e,actions)

    def test_not_allowed_pseudo_call_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=probe();e=Path(tmp)/"events";r=s2.ArmS2Events(p,e);run_call(r);r.close()
            entry=json.loads(e.read_text().splitlines()[0]);self.assertEqual(entry["call_type"],"NOT_OBSERVABLE_WITH_GENERATE_WRAPPER")

if __name__=="__main__":unittest.main()
