# Week 8 — Launch: Final Benchmarks, Docs, v0.1.0 Public

**Objective:** Ship it. Final full benchmark run, README rewritten for a 2-minute reviewer, demo assets, release hygiene, repo flipped public, pinned, and cross-linked from the profile. By Sunday night the repo is public with green CI and published numbers.

## Exit Criteria

- [ ] Final benchmark: all providers × all retrieval configs × both pipelines, guardrails on — the canonical tables
- [ ] README passes the 2-minute test (see checklist below)
- [ ] v0.1.0 tagged with release notes; repo public; pinned first on profile; profile README table updated
- [ ] A stranger following the quickstart hits zero undocumented steps (tested on a clean machine)
- [ ] Known-limitations doc published — honest scope statement

## Workstreams

### 1. Final benchmark run
- [ ] Freeze: dataset v2, prompt versions, config matrix — record all versions in results
- [ ] Full run with cost pre-estimate; judge variance handling per week 4 (3-call averaging if needed)
- [ ] Regenerate `docs/benchmarks.md`; extract headline table + 2-3 sentence findings summary for README
- [ ] Sanity-check anomalies against week 4/5 results before publishing; investigate any >0.5 rubric swings

### 2. README rewrite (the 2-minute reviewer test)
- [ ] Above the fold: one-line positioning, badge row (standard set + CI badge), architecture mermaid, headline benchmark table
- [ ] Then: quickstart (no-key Ollama path first), agentic-vs-vanilla findings, guardrails summary with red-team catch-rate table, observability screenshot, links to deep docs
- [ ] Matches the repo standard: What It Does → Setup → Usage → Architecture → License
- [ ] Demo asset: asciinema/GIF of `ask --agentic` streaming with citations; Jaeger trace screenshot
- [ ] `docs/limitations.md`: corpus scope, judge bias caveats, injection-defense honesty, what production would add (linked from README — this doc reads as senior judgment)

### 3. Release hygiene
- [ ] Dependency audit: pins current, no CVEs (`pip-audit` in CI), licenses compatible
- [ ] Repo scrub before flip: full history grep for keys/tokens (`gitleaks`), `.env.example` complete, no stray large files (`git-sizer`)
- [ ] CONTRIBUTING.md (brief), issue templates (bug/question), SECURITY.md (contact for vulnerabilities)
- [ ] Tag `v0.1.0` + GitHub release with notes: what/why/benchmark headlines; CHANGELOG.md started
- [ ] CodeQL + Dependabot enabled

### 4. Launch
- [ ] Flip public; verify badges/links render for a logged-out viewer
- [ ] Pin order update: agentic-rag first
- [ ] Profile README Open Source table: add row with one-line description
- [ ] Topics: `rag`, `agentic-ai`, `langgraph`, `llm-evaluation`, `retrieval-augmented-generation`, `guardrails`, `python`
- [ ] Optional (comfort level): short LinkedIn post with the headline finding — measured claims only

### 5. Post-launch guard
- [ ] Watch issues for quickstart failures for the first week; hotfix as v0.1.1 if needed
- [ ] Open 3-5 good-first-issue items from the limitations doc (signals live maintenance to reviewers)

## Verification

- Clean-machine quickstart executed start to finish by following README only — timed, recorded in the PR description.
- Logged-out browse: every README link resolves, images render, CI badge green.
- `gitleaks` clean on full history (private-phase commits included — plan docs are fine to ship, they demonstrate process).

## Commit Milestones (4-6 commits)

1. Final benchmark + regenerated docs
2. README rewrite + demo assets
3. Release hygiene: scrub, security files, templates
4. v0.1.0 tag + release notes
5. Post-flip fixes (topics, links, pins)

## Risks & Notes

- If any exit criterion from weeks 4-6 is unmet, **delay the flip, not the quality** — a private repo another week costs nothing; public weak numbers cost credibility.
- The plan docs in `docs/plan/` stay in history deliberately: an 8-week executed plan visible in the commit log is itself a portfolio artifact.
- After launch, this repo becomes the workload deployed by `federal-llm-blueprint` (its week 8) — keep the Docker image interface stable.
