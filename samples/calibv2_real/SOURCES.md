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
