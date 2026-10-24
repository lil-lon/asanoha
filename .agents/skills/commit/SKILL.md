---
name: commit
description: Use before creating any git commit in this repository. Runs `codex review` with the repository's review focus, fixes P0-P2 findings, then commits with a short message.
---

1. Stage only the files that belong to the change with `git add <file>` (never `-A` or `.`). The review covers committed and staged changes only.
2. Run the review with the command below, replacing `<target-branch>` with the branch this work will merge into (usually `main`). Keep the prompt on one line and do not add `--base`, `--uncommitted`, or `--commit`: they cannot be combined with a prompt.

   ```
   codex review -c review_model=gpt-6-astra "Review the changes this branch would merge into <target-branch>, limited to committed and staged changes: run git merge-base HEAD <target-branch> and inspect git diff --cached against that SHA. Ignore unstaged and untracked files. Focus on: (1) the code stays simple, with no unnecessary guards or edge-case handling. (2) numerical tests use tolerances as tight as possible and run on a variety of systems. (3) no local paths or unnecessary proper nouns, use general terms instead. (4) security vulnerabilities, such as unvalidated input, integer overflow, out-of-bounds access, and unbounded resource use."
   ```

3. If the review reports any P0, P1, or P2 finding, fix it, `git add` the fixed files, and go back to step 2. Do not commit while a P0-P2 finding remains.
4. Commit with a one-line subject and no body. Trailers such as `Co-Authored-By` are not a body.
