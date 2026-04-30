# BioSpread To 10/10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise BioSpread from a strong research-grade standalone pipeline to a stricter, more trustworthy, easier-to-maintain, and easier-to-operate production-quality ML project.

**Architecture:** Keep the current standalone package shape, but harden the project in six directions: stricter release policy, stronger validation, cleaner configuration boundaries, better testability and module structure, safer packaging/security defaults, and proper repository/process hygiene. Avoid broad rewrites; tighten behavior where the current code already has the right seams.

**Tech Stack:** Python 3.9+, setuptools, pytest, pytest-cov, mypy, ruff, scikit-learn, GitHub Actions

---

## Deep Review Summary

### What is already strong

- Cross-validated model metrics, bootstrap intervals, drift checks, trend checks, audit artifacts, and model-card output are all present.
- Test coverage is good for the current footprint and the pipeline verifies end-to-end behavior.
- CI exists and covers multi-version Python, lint, type check, coverage, dependency audit, and verification script execution.

### What currently blocks a 10/10 score

- Release policy is softer than it looks: `insufficient_data` trend status is treated as passing in [release_gate.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/release_gate.py:8).
- The main pipeline auto-switches to Geo mode when the feature surface exists, which is convenient but makes execution less explicit in [pipeline.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/pipeline.py:267).
- Type checking is permissive: `strict = false`, `ignore_missing_imports = true`, and several error classes are disabled in [pyproject.toml](/Users/umut/Projeler/bio_spread_project/pyproject.toml:40).
- CLI coverage is weak relative to the rest of the project; behavior exists but is not validated nearly as hard as core pipeline behavior.
- Configuration is split between actively used files and apparently unused files under `project_config/config/`, which increases entropy and weakens operator trust.
- `requirements.txt` mixes runtime and dev tools, uses loose lower bounds only, and relies on ignored audit entries.
- The workspace is not a Git repository, which undercuts change traceability, release provenance, and review discipline.

## File Map

### Core runtime files

- Modify: [src/bio_spread_project/pipeline.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/pipeline.py)
- Modify: [src/bio_spread_project/release_gate.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/release_gate.py)
- Modify: [src/bio_spread_project/geo_reliability.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/geo_reliability.py)
- Modify: [src/bio_spread_project/cli.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/cli.py)
- Modify: [src/bio_spread_project/audit.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/audit.py)
- Modify: [src/bio_spread_project/quality.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/quality.py)
- Modify: [src/bio_spread_project/trend.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/trend.py)
- Modify: [src/bio_spread_project/config.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/config.py)

### Tests and verification

- Modify: [tests/test_bio_spread.py](/Users/umut/Projeler/bio_spread_project/tests/test_bio_spread.py)
- Create: `tests/test_cli.py`
- Create: `tests/test_release_gate.py`
- Create: `tests/test_quality.py`
- Create: `tests/test_trend.py`
- Modify: [verify_project.py](/Users/umut/Projeler/bio_spread_project/verify_project.py)

### Packaging and CI

- Modify: [pyproject.toml](/Users/umut/Projeler/bio_spread_project/pyproject.toml)
- Modify: [requirements.txt](/Users/umut/Projeler/bio_spread_project/requirements.txt)
- Create: `requirements-dev.txt`
- Create: `constraints.txt`
- Modify: [.github/workflows/ci.yml](/Users/umut/Projeler/bio_spread_project/.github/workflows/ci.yml)
- Modify: [Makefile](/Users/umut/Projeler/bio_spread_project/Makefile)

### Documentation and governance

- Modify: [README.md](/Users/umut/Projeler/bio_spread_project/README.md)
- Modify: [docs/validation_strategy.md](/Users/umut/Projeler/bio_spread_project/docs/validation_strategy.md)
- Modify: [docs/operations_runbook.md](/Users/umut/Projeler/bio_spread_project/docs/operations_runbook.md)
- Create: `docs/architecture.md`
- Create: `docs/data_contracts.md`
- Create: `docs/release_policy.md`

### Configuration cleanup

- Modify or remove after validation: `project_config/config/*`
- Modify: [project_config/quality_thresholds.json](/Users/umut/Projeler/bio_spread_project/project_config/quality_thresholds.json)
- Modify: [project_config/trend_thresholds.json](/Users/umut/Projeler/bio_spread_project/project_config/trend_thresholds.json)
- Modify: [project_config/drift_thresholds.json](/Users/umut/Projeler/bio_spread_project/project_config/drift_thresholds.json)
- Modify: [project_config/security_audit_ignore.txt](/Users/umut/Projeler/bio_spread_project/project_config/security_audit_ignore.txt)

---

### Task 1: Make Release Gating Honest

**Files:**
- Modify: [src/bio_spread_project/release_gate.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/release_gate.py)
- Modify: [src/bio_spread_project/trend.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/trend.py)
- Modify: [src/bio_spread_project/pipeline.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/pipeline.py)
- Create: `tests/test_release_gate.py`
- Modify: [docs/operations_runbook.md](/Users/umut/Projeler/bio_spread_project/docs/operations_runbook.md)
- Modify: [docs/validation_strategy.md](/Users/umut/Projeler/bio_spread_project/docs/validation_strategy.md)

- [ ] Add an explicit policy for `trend_status in {"ok", "insufficient_data", "missing"}` instead of silently treating non-`ok` as pass.
- [ ] Introduce a configurable minimum registry history rule:
  `required_entries_for_go = 20` by default for `window_size=10`.
- [ ] Fail `release_gate.json` for scheduled or production-intended runs when trend evidence is insufficient, while allowing a documented CI/dev override.
- [ ] Add tests covering:
  `ok + pass -> go`, `ok + fail -> no_go`, `insufficient_data in strict mode -> no_go`, `insufficient_data in dev mode -> go_with_warning or explicit downgraded readiness`.
- [ ] Update the runbook so operators know exactly when a run is blocked for insufficient history versus degraded model quality.

**Acceptance criteria:**
- A run cannot score `100/go` while the trend monitor has not yet accumulated enough evidence.
- `release_gate.json` exposes whether the gate is blocked by `quality`, `drift`, or `trend_evidence`.

**Test commands:**
- `python3 -m pytest -q tests/test_release_gate.py`
- `python3 -m pytest -q tests/test_bio_spread.py -k trend`

---

### Task 2: Remove Ambiguous Runtime Behavior

**Files:**
- Modify: [src/bio_spread_project/pipeline.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/pipeline.py)
- Modify: [src/bio_spread_project/cli.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/cli.py)
- Modify: [run_project.py](/Users/umut/Projeler/bio_spread_project/run_project.py)
- Create: `tests/test_cli.py`
- Modify: [README.md](/Users/umut/Projeler/bio_spread_project/README.md)

- [ ] Replace the current `auto` behavior that silently prefers Geo surface when present with an explicit execution policy:
  `auto` may remain, but it must print and record exactly why a mode was chosen.
- [ ] Add a strict `--mode auto --require-explicit-surface` or equivalent policy flag for production usage so hidden fallback/promotion paths disappear.
- [ ] Emit selected input mode, selection reason, and all candidate input paths in the manifest and CLI stdout.
- [ ] Add CLI tests for:
  explicit `raw`, explicit `geo`, `auto` with Geo present, `auto` with Geo absent, and missing-path failures.

**Acceptance criteria:**
- No production-intended run can silently change modeling surface without leaving an explicit trace in output and manifest.
- CLI behavior is tested directly rather than only through pipeline internals.

**Test commands:**
- `python3 -m pytest -q tests/test_cli.py`
- `python3 -m pytest -q tests/test_bio_spread.py -k "missing_records or geo_mode"`

---

### Task 3: Harden ML Validation and Benchmark Semantics

**Files:**
- Modify: [src/bio_spread_project/geo_reliability.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/geo_reliability.py)
- Modify: [src/bio_spread_project/evaluation.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/evaluation.py)
- Modify: [src/bio_spread_project/audit.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/audit.py)
- Modify: [src/bio_spread_project/quality.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/quality.py)
- Modify: [project_config/quality_thresholds.json](/Users/umut/Projeler/bio_spread_project/project_config/quality_thresholds.json)
- Modify: [docs/validation_strategy.md](/Users/umut/Projeler/bio_spread_project/docs/validation_strategy.md)

- [ ] Split metric reporting into clearly named tracks:
  `oof`, `group_oof`, `temporal_holdout`, `external_holdout`, and `train_fit` if retained.
- [ ] Ensure audit and model card never imply an external or temporal benchmark exists when the metric was actually backfilled from OOF values.
- [ ] Add minimum sample-size thresholds for temporal/group/external reporting; below threshold, report `not_evaluated` instead of reusing another metric.
- [ ] Expand leakage/adversarial tests to cover contaminated feature names and contaminated values separately.
- [ ] Consider stronger robustness metrics:
  calibration by confidence tier, top-decile lift, and error breakdown by low-knownness cases.

**Acceptance criteria:**
- Every metric artifact says whether a number was actually evaluated or only unavailable.
- Quality gates do not “pass by substitution”.

**Test commands:**
- `python3 -m pytest -q tests/test_bio_spread.py -k "leakage or external_holdout or temporal"`

---

### Task 4: Tighten Types, Module Boundaries, and Maintainability

**Files:**
- Modify: [pyproject.toml](/Users/umut/Projeler/bio_spread_project/pyproject.toml)
- Modify: [src/bio_spread_project/pipeline.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/pipeline.py)
- Modify: [src/bio_spread_project/geo_reliability.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/geo_reliability.py)
- Create: `src/bio_spread_project/runtime_policy.py`
- Create: `src/bio_spread_project/manifest.py`
- Create: `src/bio_spread_project/input_selection.py`
- Create: `src/bio_spread_project/model_metrics.py`

- [ ] Refactor `pipeline.py` by moving input-selection logic, manifest assembly, and policy enforcement into focused modules.
- [ ] Introduce typed payload objects for audit/manifest/release gate instead of loosely structured `dict[str, Any]` everywhere.
- [ ] Raise mypy strictness incrementally:
  first remove disabled error classes in newly extracted modules, then narrow `Any` usage in the old files.
- [ ] Add targeted tests for the new extracted pure functions to reduce the current over-reliance on one large integration test file.

**Acceptance criteria:**
- `pipeline.py` stops being the project’s control-plane dumping ground.
- Mypy config becomes stricter without breaking the current runtime contract.

**Test commands:**
- `python3 -m mypy src`
- `python3 -m pytest -q tests`

---

### Task 5: Clean Up Configuration Debt

**Files:**
- Modify: [src/bio_spread_project/config.py](/Users/umut/Projeler/bio_spread_project/src/bio_spread_project/config.py)
- Modify: [README.md](/Users/umut/Projeler/bio_spread_project/README.md)
- Create: `docs/data_contracts.md`
- Create: `docs/architecture.md`
- Modify or remove after usage audit: `project_config/config/*`

- [ ] Inventory every file under `project_config/config/` and classify each one as:
  `used by runtime`, `used by docs only`, or `dead`.
- [ ] Either wire genuinely important config into runtime or remove it from the standalone project.
- [ ] If model weights remain built-in, document that clearly and avoid pretending there is external runtime configuration.
- [ ] Add a single architecture/data-contract document that explains raw inputs, Geo surface inputs, artifacts, and threshold files.

**Acceptance criteria:**
- There is no “mystery config directory”.
- A new engineer can tell which config files matter by reading one document and one runtime module.

**Verification commands:**
- `rg -n "project_config/config|load_project_config|ModelSpec" src README.md docs tests`

---

### Task 6: Strengthen Test Strategy and CI Signal

**Files:**
- Modify: [tests/test_bio_spread.py](/Users/umut/Projeler/bio_spread_project/tests/test_bio_spread.py)
- Create: `tests/test_cli.py`
- Create: `tests/test_release_gate.py`
- Create: `tests/test_quality.py`
- Create: `tests/test_trend.py`
- Modify: [.github/workflows/ci.yml](/Users/umut/Projeler/bio_spread_project/.github/workflows/ci.yml)
- Modify: [Makefile](/Users/umut/Projeler/bio_spread_project/Makefile)

- [ ] Break the monolithic test file into domain-focused test modules.
- [ ] Raise coverage floor from `85` toward `90` after CLI and policy tests land.
- [ ] Add a CI step that runs the CLI smoke path directly:
  `python -m bio_spread_project.cli run --mode raw ...` with fixture data.
- [ ] Add failure-mode tests for malformed CSV/TSV, empty surfaces, and invalid threshold payloads across all contract loaders.
- [ ] Add a fast path and a full path in CI instead of one broad quality job only.

**Acceptance criteria:**
- CI failures localize the broken subsystem quickly.
- CLI behavior is first-class tested behavior, not incidental coverage.

**Test commands:**
- `make test`
- `make test-cov`
- `python3 verify_project.py --skip-run-if-data-missing`

---

### Task 7: Fix Packaging, Dependency, and Security Hygiene

**Files:**
- Modify: [requirements.txt](/Users/umut/Projeler/bio_spread_project/requirements.txt)
- Create: `requirements-dev.txt`
- Create: `constraints.txt`
- Modify: [pyproject.toml](/Users/umut/Projeler/bio_spread_project/pyproject.toml)
- Modify: [project_config/security_audit_ignore.txt](/Users/umut/Projeler/bio_spread_project/project_config/security_audit_ignore.txt)
- Modify: [.github/workflows/ci.yml](/Users/umut/Projeler/bio_spread_project/.github/workflows/ci.yml)

- [ ] Separate runtime and dev dependencies.
- [ ] Pin or constrain transitive dependencies tightly enough that verification is reproducible across CI and local runs.
- [ ] Document every ignored vulnerability with package, impact, reason, and removal target date.
- [ ] Add an install path that uses the package itself:
  `pip install -e .[dev]` or equivalent, instead of ad hoc runtime-only dependency installs.
- [ ] Consider adding `pip-audit` to check the installed environment rather than only `requirements.txt`.

**Acceptance criteria:**
- A fresh machine can reproduce installs predictably.
- Vulnerability ignores are explicit debt, not silent wallpaper.

**Verification commands:**
- `python3 -m pip_audit -r requirements.txt`
- `python3 -m pip install -e .[dev]`

---

### Task 8: Add Repository and Release Discipline

**Files:**
- Create: `.git/` repository metadata by initializing the project as a real Git repository
- Create: `.gitattributes`
- Create: `.editorconfig`
- Modify: `.gitignore`
- Create: `CHANGELOG.md`
- Create: `CONTRIBUTING.md`
- Create: `docs/release_policy.md`

- [ ] Initialize this standalone project as its own Git repository if that is truly intended.
- [ ] Add basic repository hygiene for line endings, generated artifact rules, and contributor workflow.
- [ ] Decide which `reports/` and `data/` artifacts belong in version control versus release assets only.
- [ ] Define release semantics:
  what counts as a model release, what requires benchmark refresh, what requires threshold review.

**Acceptance criteria:**
- Provenance no longer depends on an external parent repo or manual memory.
- The standalone extraction behaves like a maintained project rather than a copied folder snapshot.

**Verification commands:**
- `git status`
- `git log --oneline`

---

### Task 9: Improve Operator-Facing Documentation and Trust Surface

**Files:**
- Modify: [README.md](/Users/umut/Projeler/bio_spread_project/README.md)
- Modify: [docs/operations_runbook.md](/Users/umut/Projeler/bio_spread_project/docs/operations_runbook.md)
- Modify: [docs/validation_strategy.md](/Users/umut/Projeler/bio_spread_project/docs/validation_strategy.md)
- Create: `docs/architecture.md`
- Create: `docs/data_contracts.md`
- Create: `docs/release_policy.md`

- [ ] Reduce optimistic wording and make every claim map to a verifiable artifact or test.
- [ ] Document what “standalone” means operationally:
  packaged data assumptions, storage expectations, compute expectations, and model refresh procedure.
- [ ] Add one artifact glossary so `audit.json`, `benchmark.json`, `manifest.json`, `release_gate.json`, and `model_registry.jsonl` each have one owner-purpose paragraph.

**Acceptance criteria:**
- Operators do not need to read source code to understand safe usage and limitations.
- README becomes accurate, shorter, and less promotional.

---

## Recommended Order

1. Task 1: release-gate honesty
2. Task 2: explicit runtime behavior
3. Task 3: validation semantics
4. Task 6: tests and CI
5. Task 4: modularity and typing
6. Task 7: packaging/security
7. Task 5: config cleanup
8. Task 9: documentation
9. Task 8: repository discipline

## Suggested Milestones

- **Milestone A: Trustworthy gating**
  Tasks 1, 2, 3, and the critical parts of Task 6
- **Milestone B: Maintainable codebase**
  Task 4 and Task 5
- **Milestone C: Operable project**
  Task 7, Task 8, and Task 9

## Definition of “Near 10/10”

- No silent mode switching in production-intended runs
- No false-green `go` status under missing trend evidence
- Validation artifacts distinguish measured vs unavailable metrics
- Coverage floor at or above `90` with CLI and policy tests included
- Stricter mypy posture with reduced `Any` spread
- Runtime and dev dependencies separated and reproducible
- Unused config removed or wired into runtime
- Project runs inside a proper Git lifecycle with documented release policy

## Immediate Highest-Leverage First Sprint

- Tighten release gate semantics
- Make mode selection explicit and traceable
- Add dedicated CLI and release-policy tests
- Raise coverage floor only after those tests land

Plan complete and saved to `docs/superpowers/plans/2026-04-26-biospread-to-10.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
