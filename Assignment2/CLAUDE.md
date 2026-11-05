# CLAUDE.md

## Project

IACV Assignment #2 — Object Recognition and Image Retrieval (group of 2). Due **2026-11-08**.
Handcrafted (BoVW/VLAD/FV) → pretrained (ResNet-18, ViT, DINOv2) → CLIP zero-shot, plus a retrieval challenge.
Brief: `Assignment_2_Object_Recognition_and_Retrieval.pdf`. Repo README has data/feature details.
Full plan and phase checkpoints: `assignment_plan.md`.

Protocol (applies to every phase):
- Fixed train/val/test splits, identical across methods. Test set used **once**, for final numbers only.
- Hyperparameters, prompts, C, K chosen on validation only.
- Small-data runs: ≥3 seeds, report mean ± std.
- Log per method: AP/mAP, feature dim, extraction time, inference cost, storage.

### Phases

1. Setup (env pinned, data + precomputed BoVW/VLAD/FV loaded, splits fixed, shared metrics module: PR curve, AP, mAP, P@K, R@K, confusion matrix) → sanity check on precomputed BoVW
2. Part 1 — BoVW (global vs 2×2 tiling, K = 256/512/1024, motorbike linear SVM, ranking + relevant visual words, PR/AP on test)
3. Part 2 — VLAD + Fisher Vector (same splits/classifier, compare vs BoVW: AP, dim, cost)
4. Part 3 — Large vocabulary + PCA vs small vocabulary
5. Part 4 — Pretrained features (ResNet-18 last 3 layers GAP, ViT, DINOv2 ViT-S/14 + SVM; few-shot 1/5/10/20 + flip aug; C sweep; CLIP zero-shot + prompt sets; failure cases + CAM/patch similarity)
6. Part 5 — Retrieval challenge (corpus + queries + ground truth, ≥2 strategies, P@5/10/20, AP/mAP, cost)
7. Report + polish (comparison table, plots, discussion answers, README, reproduce scripts) 

## Model workflow

| Role | Model | Used for |
| --- | --- | --- |
| Builder | Sonnet (main session) | Reads scouted files, edits, runs smoke tests |
| Scouts | Haiku subagents | Read-only searches, scoped to the current phase |
| Advisor | Opus (`/advisor`) | Decision points only; never writes code |

Start with `claude --model sonnet`, then enable the advisor once with `/advisor`.

### Scouts (Haiku)

Use `caveman:cavecrew-investigator` or the built-in `Explore` agent (both Haiku) for:

- Mapping a part of the repo, or finding all call sites of a symbol
- Delta inventory items: return paths, line numbers, and ARM-specific flags only
- Repo-wide audits against a rule
- Reading large files or logs to extract a few facts

Rules:

- Read-only. Scouts never edit.
- Return a short summary, not file contents.
- Run independent searches in parallel.
- Single known file? Just read it, no scout.
- `caveman:cavecrew-reviewer` (Haiku) is for first-pass checks only. Sonnet does the final review.
- Do not use `caveman:cavecrew-builder` (inherits the main model). All edits stay in the main session.

### Builder (Sonnet)

- All edits and test runs happen in the main session.
- Scout first, then read only the scouted files.
- Small, targeted edits. Run the phase smoke test before starting the next phase.
- Diffs: caveman + ponytail (compress commits, skip speculative features, stdlib first).

### Advisor (Opus)

Call `/advisor` only:

- Between major groups of phases, not between every phase
- When stuck: same error twice, or a fix that changed nothing
- Before declaring the whole port done

Never mid-phase for routine work. Each call reads the full transcript, so it is expensive. Sonnet applies the advice itself.

---

## Context hygiene

- `/clear` after a phase's smoke test passes.
- `/compact` if the transcript passes ~50 turns mid-phase.
- At each phase end, record in memory: phase done, and what blocks the next one.
- Keep this file short. It loads every session.
- If the advisor costs more than it saves: `/advisor off`.
