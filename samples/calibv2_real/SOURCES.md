# Real fragments for the calibration v2 judge tests

Byte-identical copies (`cp -p`, 2026-11-15) of read-only files under the work dir
`W = /home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02`, used by
`tests/judge/test_calibv2_sources.py` (calib-v2 DESIGN-r7a 3.5 tests 17 and 25). The tests
check each sha below before using the copy.

| File | Source (relative to W) | sha256 |
|---|---|---|
| `e1_0_http.register.json` | `03_calib/E1/register/http/register.json` (E1-0 register; its `calibration` names the round-1 pilot) | `242f52af1ba44c2b271e4dc795054b26c986375e51f9af9ceea8c4fc927131b4` |
| `e0_round1.pilot.json` | `02_e0/E0/analysis/pilot.json` (`capkit.pilot/1`; REGISTRY.tsv line 43, kind `pilot`) | `8e823f128a3c6104c3c6f4fdd8931c148f695ed9f2f8126bdc38b2b4464ea39a` |
| `e0_round1.register.json` | `02_e0/E0/register/register.json` (E0 round-1 register, no `arm_config`) | `7c0f268cc8431bbcc7358f3f081a3c2d0803594355ca74a22ba3ce9bc60f4fa8` |

`j5940628_l1_t4_requests.json` (sha256 `92dbf6ec983246fc687f3daf7e97f934d0df23e53a8803ce4d2f3c9beef79f36`, used by
`tests/judge/test_judge_levels.py`, calib-v2 DESIGN-r7a 3.5 test 6) is an extract, not a copy:
the `planned_rel_ns`, `first_rel_ns` and `status` columns of the capload request records
(workers 0 and 1, rows planned in the steady window [140 s, 200 s) of tier 4 `up-sub-hi`,
100 rps) of the run tree `02_e0/E0/runs/w60/J5940628-r0-L1_e0a-nullsse-w60` (e0a-nullsse-w60,
the two-level switching case of calib-v2 DESIGN §1.5), read with `capcommon.capbin.read_capbin`
from

| Source file (relative to W) | sha256 |
|---|---|
| `02_e0/E0/runs/w60/J5940628-r0-L1_e0a-nullsse-w60/raw/tiers/004_up-sub-hi/capload.w0.requests.capbin` | `03879caa60f43ae2f24fd6549c3fe932fb4bead8bed04d110d8b7c141301bfe6` |
| `02_e0/E0/runs/w60/J5940628-r0-L1_e0a-nullsse-w60/raw/tiers/004_up-sub-hi/capload.w1.requests.capbin` | `585547bf8cdd64c1e19ee0b6d74adf9547271bff58d4785d36ac8907fef7d46f` |

`plan_r4_S_fit_by_K.json` (used by `tests/launch/test_laddergen.py`, calib-v2 DESIGN-r7a 3.5
test 12) is an extract, not a copy: the keys `S_fit_by_K`, `table_max_s` and
`job_overhead_s_max_floor` of `02_e0/E0/analysis/calib-v2/r4/out/plan_r4.json` (sha256
`551548b73e937c83ef4fe2d3b838fcc9aa9e23133ad1b8ae0964d097b38b6366`; written by `r4/plan_r4.py`),
re-serialised with `json.dump(..., indent=1, sort_keys=True)`.

`e0_round1.pilot_inputs.tsv` (used by `tests/judge/test_pilot2.py`, calib-v2 DESIGN-r7a 3.5
tests 27 and 34) is a byte-identical copy (`cp -p`, 2026-11-15) of `02_e0/E0/analysis/pilot_inputs.tsv`
(the 15 round-1 trees and tiers the registered `pilot.json` was built from), sha256
`344e0c45fce623aafb670e1aa3a66368b15d0a20754edd8fb8e0173a02c0fd24`. The tests read the trees
it names and the registered D14 rule `02_e0/E0/register/d14_rule.json` (sha256
`ae28dec44bc0579dd34e9774c64569a86a6bcbfdef135b46885c5cce204d8444`, REGISTRY.tsv line 2) in
place, read only, and are skipped where W is absent.
