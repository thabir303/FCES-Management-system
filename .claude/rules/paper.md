---
paths:
  - "main.tex"
---

# Rules for the paper

**`main.tex` is the paper and the authority.** Anything it claims, the code must deliver. If the
plan and the paper disagree, stop and raise it — do not choose a side.

**`make paper` must build clean before any commit touching `main.tex`** — two `pdflatex` passes, no
`bibtex` (the bibliography is inline; there is no `.bib`). TeX is at `/Library/TeX/texbin`, not on
the default PATH. A precondition exactly as a clean tree is for `make tables`.

**No number is ever typed in by hand.** Every figure is `\input{}` from `results/tables/`. A number
no run produced does not belong in the paper, however confident you are of it.

**Never remove a `[TBC]` by writing in a value you have not measured.** If a marker count moves the
wrong way, stop and check.

**Four rules on editing, from the supervisor:**

1. Correcting a statement of fact that a measurement contradicts **is** yours to do. Make the
   change, then say what changed and which run contradicted it.
2. Changing what the paper **claims, promises or argues** is not yours. Propose and wait. The test:
   does the sentence state something measurable that turned out false, or something the study
   intends to do?
3. **Never adjust a claim to accommodate a disappointing result.** A disappointing result is
   reported as it stands. A claim is corrected only when the claim itself was wrong — never because
   a number came out lower than hoped. If you find yourself softening a sentence so a result reads
   better, stop and raise it.
4. **Never remove or weaken a threat to validity.**

**Mechanical properties to preserve:** no comments, no instructions to a reader, and bare
unescaped `[TBC]` as the only placeholders. **The paper gets its own commits** — sweeping a paper
change in with code has already left author and supervisor reading different states.

**Verify before overwriting.** Replacements arrive with a fingerprint (line and byte counts, `[TBC]`
counts, required and forbidden phrases). Check all of it. `grep -F` misses phrases spanning a line
break in this hard-wrapped file — collapse whitespace before matching.

**Stop on any mismatch you cannot fully explain. You may proceed on one you can prove**, provided
the fix makes the **file** match the spec rather than making the **spec** match the file, and
provided you say what you did.

That proviso is the whole safeguard, so read it precisely. A missing terminal newline explaining a
discrepancy of exactly one line and exactly one byte, with every content check already passing, is
provably not a wrong file: restoring the newline so the file becomes the specified length is the
right response. Editing the stated count down to match the file is the same observation and the
wrong response, because after it nothing checks anything. Directional checks are what tell a newer
file from an older one — a `[TBC]` count that has fallen is evidence you have the newer file, and a
count that has *risen* is a mismatch you cannot explain, so it stops you. A fingerprint exists to
catch a wrong file, not to be satisfied.
