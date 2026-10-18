---
name: commit
description: Use before creating any git commit in this repository. Runs `codex review` with the repository's review focus, fixes P0-P2 findings, then commits with a short message.
---

1. Make sure the working tree contains only the change to be committed, and `git add` any new files so the review sees them.
2. Run the review with the command below, replacing `<target-branch>` with the branch this work will merge into (usually `main`). Keep the prompt on one line and do not add `--base`, `--uncommitted`, or `--commit`: they cannot be combined with a prompt.

   ```
   codex review "Review the changes this branch would merge into <target-branch>, including uncommitted changes: run git merge-base HEAD <target-branch> and inspect git diff against that SHA. Focus on: (1) the code stays simple, with no unnecessary guards or edge-case handling. (2) numerical tests use tolerances as tight as possible and run on a variety of systems. (3) no local paths or unnecessary proper nouns, use general terms instead."
   ```

3. If the review reports any P0, P1, or P2 finding, fix it and go back to step 2. Do not commit while a P0-P2 finding remains.
4. Commit with a one-line subject and a body of zero or one sentence. Trailers such as `Co-Authored-By` do not count toward the body.
