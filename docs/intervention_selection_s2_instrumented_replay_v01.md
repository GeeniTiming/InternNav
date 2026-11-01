# v0.1 passive System-2 call capture without changing frozen source code

The existing `intervention_selection_paired_v01.py` and `intervention_selection_perturbed_keep_v01.py` scripts are SHA256-pinned by previous experiments. **Do not modify them or their old results**. The new `intervention_selection_s2_replay_v01.py` calls their identical `PairProbe` / `PerturbedKeepProbe` logic, but intercepts each actual System-2 `model.generate` invocation and writes a create-only `s2_call_events.jsonl` with paired enter/exit records in a **new** arm directory. An exact action offset is observed, never inferred from `main_step`. The original evaluation model, checkpoint replay, scan and history-control code is reused.

`ArmS2Events` also stores the decoded output returned for this exact call and a hash of its textual token input. It does not capture RGB pixel values or prove an arm-to-arm intervention effect; gate hashes, physical scan/frame audits and pre-arm equality must still pass.

Only **fresh prepared v0.1 runs** can be used. To launch new paired/perturbed KEEP source arms (GPU required), the commands are:

```
python scripts/eval/intervention_selection_s2_replay_v01.py run-perturbed-keep --run-dir /path/to/FRESH_PREPARED_PERTURBED_KEEP
python scripts/eval/intervention_selection_s2_replay_v01.py run-paired --run-dir /path/to/FRESH_PREPARED_PAIRED --arm perturbed_seek_right45
```

**Do not run these against `intervention_v01_perturbed_keep_01` or `intervention_v01_paired_all10_01`.** Those arms already exist; the new writer correctly refuses overwrites. Existing old native logs are unchanged and do not gain missing S2 calls.

The original v0.1 prepare protocol permits **first 2 or all 10 eligible** episodes. It does not permit selective 3-case replay, and the old arm vocabulary does not include `SCAN_NO_INJECTION` or `PLACEBO`. Consequently these commands are **not** the requested 12-arm 3-case mechanism ablation. The next implementation step is to create a distinctly named, posthoc mechanism-only protocol selecting exactly Episode 70, 106 and 79, with four matched arms each, and add the call capture to every one. No generalization claims are possible from posthoc-selected cases.

## CPU smoke tests

```
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test -p 'test_intervention_selection_s2_call_events_v01.py' -v
```

8 tests pass in the local CPU harness. The new replay entrypoint has been syntax-checked, **not** Habitat GPU-tested. All new arm reports remain absent until a real fresh replay runs and verifies its own frozen gate and action prefixes.
