# Workflow log

What was tried, what it led to, and what came next. Updated every week. The picture version is in the shared status doc.
Rule: add a weekly entry before closing the week, even if the week produced nothing (say so).

## Tree (what led to what)

```
Instruct teacher pilot (k=5; student pass@1..8; k=8 extension)
  -> Gold-based answer-position cut                      RETRACTED: circular, the cut used the gold answer
  -> Reasoning teacher v1 (3072 tokens)                  74 of 250 traces had no \boxed answer (truncation)
       -> Reasoning teacher v2 (8192 tokens)             232 clean traces; 18 dropped; problem 43 fully lost
            -> Measurement audit                         matcher fixes; gold-free cut; earlier numbers retracted
                 -> Signal retention test                VALID (proxy): ~92% reasoning, ~94% instruct kept
                 -> Cut audit (30 reasoning cuts)        27 clean; owner to confirm
                 -> Problem-set check                    only 5 problems shared between the teachers
                      -> NEXT: like-for-like run on the instruct problems, then fine-tuning transfer test
```

## Weekly entries

### Week of 2026-10-12
- Audited labelling and cut logic; fixed five matcher bugs; replaced the gold-based cut with a gold-free cut.
- Reproduced final numbers for both teachers (see RESULT_LOG.md). Repo restructured and pushed.
- Found the teachers ran on different problems (5 shared). Wording corrected.
- Drafted the supervisor email (repo link, two-week break, next meeting 2026-10-29). Edit this line with what actually happened.

### Week of 2026-11-02
- Set up logging: ERROR_LOG.md, RESULT_LOG.md/.csv, WORKFLOW.md, `reporting/make_figures.py` (figures come only from files), and a Colab notebook with visible outputs.
- Shared status doc created (Claude Docs).
- Planned next: like-for-like run (step 1 of the action plan), cut confirmations, 16k regeneration.
