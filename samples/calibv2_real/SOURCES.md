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
