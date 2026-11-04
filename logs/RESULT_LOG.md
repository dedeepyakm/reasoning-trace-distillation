# Result log

Every number here names where it came from. Rule: no number goes into a slide, email or the status doc unless it has a row here.
`logs/RESULT_LOG.csv` is filled in automatically by `reporting/make_figures.py`; this file is the curated summary.

Status: **valid** (reproduced after the Oct 2026 audit), **retracted** (do not quote), **pending** (to be checked).

| Date | Result | Value | Source file | Script | Status |
|---|---|---|---|---|---|
| Oct 2026 | Reasoning teacher accuracy (49 problems, 232 clean traces) | 216/232 = 93.1% (86.4% if the 18 dropped traces count as wrong) | `data/cut_reasoning_final.jsonl` | `analysis/analyze_traces_v3.py` | valid |
| Oct 2026 | Reasoning teacher accuracy by tier (Easy / Medium / Hard) | 97.6 / 100 / 84.6% | `data/cut_reasoning_final.jsonl` | `analysis/analyze_traces_v3.py --with-difficulty` | valid |
| Oct 2026 | Reasoning cut position (before last two / last two / never) | 181 / 22 / 29 | `data/cut_reasoning_final.jsonl` | `analysis/analyze_traces_v3.py` | valid |
| Oct 2026 | Reasoning words kept (correct / incorrect) | 56.2% (N=189) / 61.4% (N=14) | `data/cut_reasoning_final.jsonl` | `analysis/analyze_traces_v3.py` | valid |
| Oct 2026 | Reasoning signal retention | 91.6% (95% interval 73-115%); separation 1.476 cut vs 1.611 full; 60 correct / 16 incorrect scored | `data/score_reasoning_final.json` | `scoring/score_cut_v2.py` | valid |
| Oct 2026 | Reasoning cut precision (30 random cuts) | 27 clean (90%, interval 74-97%); 2 hedged, 1 early; labelled by Claude | `data/audit_reasoning_v5.csv` | `diagnostics/audit_cuts.py` | pending owner confirmation |
| Oct 2026 | Instruct teacher accuracy (50 problems, 250 traces) | 184/250 = 73.6% (Easy 96.0, Medium 76.0, Hard 50.0) | `data/cut_instruct_final.jsonl` | `analysis/analyze_traces_v3.py --with-difficulty` | valid |
| Oct 2026 | Instruct cut position (before last two / last two / never) | 35 / 139 / 76 | `data/cut_instruct_final.jsonl` | `analysis/analyze_traces_v3.py` | valid |
| Oct 2026 | Instruct words kept (correct / incorrect) | 92.5% (N=132) / 87.7% (N=42) | `data/cut_instruct_final.jsonl` | `analysis/analyze_traces_v3.py` | valid |
| Oct 2026 | Instruct signal retention | 93.5% (95% interval 87-99%); separation 1.736 cut vs 1.857 full; 60 / 60 scored | `data/score_instruct_final.json` | `scoring/score_cut_v2.py` | valid |
| Oct 2026 | Problem overlap between teachers | 5 shared; 45 instruct only; 44 reasoning only; accuracy on the 5 shared: instruct 20/25, reasoning 19/23 | `data/cut_instruct_final.jsonl`, `data/cut_reasoning_final.jsonl` | `diagnostics/compare_problem_sets.py` | valid |
| Oct 2026 | Reasoning v1 truncation | 148/250 = 59.2% correct; 74 traces without `\boxed{}` (29.6%) | `data/traces_reasoning_teacher_k5.jsonl` | `diagnostics/check_truncation_reasoning.py` | valid (generation fact) |
| Oct 2026 | Reasoning v2 truncation | 26/250 hit the token limit; 18 had no answer and were dropped | `data/traces_reasoning_teacher_k5_v2.jsonl` | `diagnostics/check_truncation_reasoning.py` | valid (generation fact) |
| Oct 2026 | "Never explicit means correct only ~8%"; "~49% last-two" (instruct) | n/a | n/a | old gold-based cut | **retracted** (circular) |
| Oct 2026 | Reasoning "74.8% accuracy", "73.7% early", "45.4% words kept", "9% retention" | n/a | n/a | old extractor and cut | **retracted** (bugs) |
| Oct 2026 | Intermediate retention 78% / 112% / 94% | n/a | n/a | earlier runs | **retracted** (superseded) |
