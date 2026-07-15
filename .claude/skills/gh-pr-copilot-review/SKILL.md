---
name: gh-pr-copilot-review
description: Commit local changes, push a branch, create a GitHub pull request, request GitHub Copilot as reviewer, verify Copilot actually started through PR events, and monitor Copilot review progress. Use when the user asks to publish changes to GitHub, open a PR, request Copilot review, check Copilot review progress, or set up periodic PR review monitoring.
---

# GitHub PR Copilot Review

## Overview

Publish local work to GitHub end to end: inspect scope, branch, stage, commit, push, create a PR, request GitHub Copilot code review, verify the request through GitHub issue events, monitor Copilot review output, address substantive feedback, push fixes, re-request Copilot review, and continue until unresolved feedback is only nits or non-actionable.

Claude Code does not receive passive GitHub callbacks in an ordinary session. To wait for a review, use a background watcher that polls internally and wakes you when it finishes (see "Wait For The Review Without Foreground Polling").

## Publish Workflow

When the user invokes this skill, execute the workflow automatically when the intended change scope is clear. Ask before continuing only if the worktree contains unrelated changes, the target branch/base is ambiguous, authentication is missing, or an operation would be destructive.

1. Inspect the worktree before staging:

   ```bash
   git status -sb
   git diff --stat
   git diff
   ```

2. Stage only the intended files. Do not use broad staging when unrelated changes exist.

3. Ensure the branch is suitable:

   - If on a detached HEAD, `main`, `master`, or the default branch, create a `claude/<short-description>` branch.
   - If already on a feature branch, keep it unless the user asks for a different branch.

4. Commit with a concise message after checking the staged diff:

   ```bash
   git diff --cached --stat
   git diff --cached
   git commit -m "<message>"
   ```

5. Run relevant validation for the change when practical. For documentation-only changes, explicitly say no code tests were needed.

6. Push with upstream tracking:

   ```bash
   git push -u origin "$(git branch --show-current)"
   ```

7. Create the PR. Default to the remote default branch unless the user specifies a target:

   ```bash
   gh pr create --base <base> --head "$(git branch --show-current)" --title "<title>" --body "<body>"
   ```

8. Request Copilot review by running `scripts/copilot-review.sh request` (the GraphQL bot-review path documented below). Use it for both initial review requests and re-review requests after pushing fixes. Do not rely on `gh pr edit --add-reviewer Copilot` or `gh pr edit --add-reviewer @copilot`; those commands can return success without creating a visible pending Copilot request or a new `copilot_work_started` event.

## Helper Script

The shell-fragile steps (request, verify, status, wait) are packaged in
`scripts/copilot-review.sh`, so they run identically every session instead of
being retyped and re-broken. It has a `#!/usr/bin/env bash` shebang, so it is
immune to the caller's interactive shell — importantly it does not hit the
`[ "$a" \> "$b" ]` string compare that errors under zsh with
"condition expected: >" and silently defeats a hand-written watcher.

`owner`/`repo`/`pr` auto-detect from the current repo and checked-out PR branch
when the flags are omitted. Reference paths relative to the skill directory:

```bash
scripts/copilot-review.sh request   # request / re-request Copilot review
scripts/copilot-review.sh verify    # events + pending reviewers (confirm it registered)
scripts/copilot-review.sh status    # submitted reviews (user, state, submitted_at, body)
scripts/copilot-review.sh wait [--baseline TS] [--interval SECS] [--max-iters N]
# add --owner O --repo R --pr N to any of the above to be explicit
```

Prefer this script over the raw `gh` snippets below; the snippets document what
each subcommand does under the hood and remain the fallback if the script is
unavailable.

## Request Or Re-Request Copilot Review

Run `scripts/copilot-review.sh request` (works for both the initial request and
re-review after pushing fixes). Under the hood it uses GraphQL
`requestReviewsByLogin` with `botLogins:["copilot-pull-request-reviewer"]` — the
reliable programmatic equivalent of the web UI's Copilot review/re-review
request:

```bash
PR_ID=$(
  gh api graphql \
    -f query='query($owner:String!, $repo:String!, $number:Int!) {
      repository(owner:$owner, name:$repo) {
        pullRequest(number:$number) { id }
      }
    }' \
    -F owner="$OWNER" \
    -F repo="$REPO" \
    -F number="$PR_NUMBER" \
    --jq '.data.repository.pullRequest.id'
)

gh api graphql \
  -f query='mutation($pr:ID!) {
    requestReviewsByLogin(input:{
      pullRequestId:$pr,
      botLogins:["copilot-pull-request-reviewer"],
      union:true
    }) {
      pullRequest { id }
    }
  }' \
  -F pr="$PR_ID"
```

Notes:

- `union:true` preserves existing requested reviewers while adding Copilot.
- `Copilot` appears in REST review requests as a bot/user-like reviewer with login `Copilot`, but the GraphQL bot login to request is `copilot-pull-request-reviewer`.
- If the GraphQL bot-login request unexpectedly fails, inspect the current requested reviewer payload for the Copilot `node_id` and fall back to `requestReviews(input:{pullRequestId:$pr, botIds:["<BOT_ID>"], union:true})`.

Fallback example:

```bash
gh api graphql \
  -f query='mutation($pr:ID!, $bot:ID!) {
    requestReviews(input:{
      pullRequestId:$pr,
      botIds:[$bot],
      union:true
    }) {
      pullRequest { id }
    }
  }' \
  -F pr="$PR_ID" \
  -F bot="<copilot-bot-node-id>"
```

## Verify Copilot Request

Run `scripts/copilot-review.sh verify`. It reports issue events and pending
requested reviewers; GitHub may leave `reviewRequests` empty in `gh pr view`,
and submitted Copilot reviews remove Copilot from the pending requested-reviewer
list, so events are the reliable signal. Equivalent raw calls:

```bash
gh api repos/<owner>/<repo>/issues/<pr-number>/events --paginate \
  --jq '.[] | select(.event=="review_requested" or .event=="copilot_work_started") | {event, actor:.actor.login, requested:(.requested_reviewer.login // null), app:(.performed_via_github_app.slug // null), created_at}'

gh api repos/<owner>/<repo>/pulls/<pr-number>/requested_reviewers

gh api repos/<owner>/<repo>/pulls/<pr-number>/reviews \
  --jq '.[] | {user:.user.login, state, submitted_at}'
```

Success signals:

- A fresh `review_requested` event with `requested_reviewer.login: Copilot`.
- A fresh `copilot_work_started` event with `performed_via_github_app.slug: copilot-pull-request-reviewer`.
- A pending requested reviewer with login `Copilot`.
- A new submitted review from `copilot-pull-request-reviewer[bot]` or `Copilot` after the latest commit.

## Monitor Copilot Review

Check progress by reading reviews, review comments, issue comments, status checks, and issue events:

```bash
gh pr view <pr-number-or-url> --json latestReviews,comments,reviewDecision,statusCheckRollup,mergeStateStatus,reviewRequests,baseRefName,headRefName
gh api repos/<owner>/<repo>/pulls/<pr-number>/reviews
gh api repos/<owner>/<repo>/pulls/<pr-number>/comments
gh api repos/<owner>/<repo>/issues/<pr-number>/events --paginate
```

Treat Copilot as done when it submits a review from `copilot-pull-request-reviewer[bot]` or `Copilot`. Report:

- review state and submitted time
- summary review body
- inline comments with file paths and comment URLs
- CI status
- whether the branch is behind the base branch

### Wait For The Review Without Foreground Polling

Copilot reviews take a couple of minutes. Do not sit in a foreground loop
re-running `gh` by hand — there are no passive GitHub push callbacks, but you
can make the wait hands-off with a single background watcher that polls
internally and exits the moment a review lands. When a `run_in_background`
command finishes, the harness re-invokes you automatically, so this behaves
like "notify me when the review is ready" while you stay idle in between.

Launch the `wait` subcommand with `run_in_background: true`:

```bash
# First round — empty baseline, so any Copilot review matches:
scripts/copilot-review.sh wait

# Re-review round — pass the previous round's latest Copilot submitted_at so it
# waits for the NEW review instead of returning the old one immediately:
scripts/copilot-review.sh wait --baseline 2026-07-15T15:58:45Z
```

It prints `COPILOT_REVIEW_READY latest=<ts>` and exits 0 the moment a matching
review lands, or `WATCHER_TIMEOUT_NO_NEW_REVIEW` after `--max-iters` polls
(default 60 × 20s = 20 min). Each round gets its own watcher.

Do NOT reimplement this poll loop inline. The earlier inline version used
`[ "$LATEST" \> "$BASELINE" ]`, which errors under zsh ("condition expected: >")
and never matches — the watcher then loops uselessly until timeout while the
review sits ready. The script avoids this with a bash shebang and `[[ > ]]`.

## Keep The PR Branch Updated

While monitoring or addressing a PR, keep the PR branch current with the base branch. This is the local equivalent of clicking GitHub's "Update branch" button.

When `gh pr view` reports that the branch is behind, the merge state is blocked by needing an update, or new commits land on the base branch during review:

1. Fetch the base branch:

   ```bash
   git fetch origin <base-branch>
   ```

2. Merge the fetched base branch into the PR branch:

   ```bash
   git merge origin/<base-branch>
   ```

3. If there are merge conflicts, resolve them locally. Do not leave conflict markers. After resolving conflicts:

   ```bash
   git status -sb
   git diff
   git add <resolved-files>
   git commit
   ```

   Use the merge commit message unless a clearer conflict-resolution message is needed.

4. Run relevant validation after the merge or conflict resolution.

5. Push the updated PR branch:

   ```bash
   git push
   ```

6. Wait for CI to run again on the updated branch. Monitor the new check run until it finishes. If CI fails, inspect logs, fix the failure, push again, and keep monitoring.

7. If the update or conflict resolution changes code that Copilot previously reviewed, re-request Copilot review with `scripts/copilot-review.sh request`.

## Address Review Feedback

When Copilot or another reviewer leaves comments, keep working until substantive feedback is addressed and review threads are resolved. Do not stop after replying to comments if code changes are still needed.

When a review comment has been addressed, resolve the corresponding GitHub review thread yourself. A reply saying "fixed" is not enough; the thread should be marked resolved through GraphQL or the GitHub UI unless it is intentionally left open as a nit, disagreement, or follow-up decision for the user.

1. Read reviews, flat comments, and thread-aware state:

   ```bash
   gh api repos/<owner>/<repo>/pulls/<pr-number>/reviews
   gh api repos/<owner>/<repo>/pulls/<pr-number>/comments
   gh api graphql -f query='query($owner:String!, $repo:String!, $number:Int!) {
     repository(owner:$owner, name:$repo) {
       pullRequest(number:$number) {
         reviewThreads(first:100) {
           nodes {
             id
             isResolved
             isOutdated
             comments(first:50) {
               nodes {
                 id
                 databaseId
                 author { login }
                 body
                 path
                 line
                 url
               }
             }
           }
         }
       }
     }
   }' -F owner=<owner> -F repo=<repo> -F number=<pr-number>
   ```

2. Classify each unresolved thread:

   - **Actionable:** correctness, regression, missing test, broken behavior, security, performance, or maintainability issue.
   - **Nit:** style preference, naming preference, wording tweak, optional micro-refactor, or low-risk polish.
   - **Non-actionable:** duplicate, already fixed by later commits, outdated, incorrect, or informational.

3. Implement all actionable feedback. For nits, either fix them when cheap or leave a concise reply explaining why they are optional.

4. Run relevant validation, commit, and push fixes.

5. Reply to each addressed thread with the fix commit or reasoning, then resolve threads that are fixed, outdated, duplicate, or non-actionable. Do this yourself as part of the review loop:

   ```bash
   gh api graphql \
     -f query='mutation($thread:ID!) {
       resolveReviewThread(input:{threadId:$thread}) {
         thread { id isResolved }
       }
     }' \
     -F thread=<review-thread-id>
   ```

6. Re-request Copilot review with `scripts/copilot-review.sh request` after every push that addresses Copilot feedback.

7. Keep the branch updated with the base branch while this loop is running. If the base branch changes, merge it into the PR branch, resolve any conflicts, push, and wait for CI to rerun.

8. Repeat monitoring, updating from base, fixing, pushing, replying, resolving, and re-requesting until:

   - all CI checks pass,
   - merge state is clean or mergeable,
   - the branch is not behind the base branch,
   - there are no unresolved actionable threads,
   - any remaining unresolved comments are clearly nits or non-actionable,
   - Copilot has either submitted a post-fix review or the latest re-request is visibly pending/started.

## Periodic Follow-Up

If the user asks to watch, monitor, check back, notify them, or get callbacks, use the background watcher in "Wait For The Review Without Foreground Polling" rather than claiming passive callbacks exist. It polls GitHub internally and wakes you on completion, so you never sit in a foreground loop. Report only meaningful changes when it fires: new Copilot review, new comments, failed checks, or merge readiness.

## Important Details

- Prefer `scripts/copilot-review.sh` (subcommands `request`, `verify`, `status`, `wait`) over hand-typed `gh`/loop snippets — it is the tested, shell-portable entrypoint. Reach for raw `gh` only to debug or when the script is unavailable.
- Never hand-roll the review-wait poll loop in the interactive shell: a `[ "$a" \> "$b" ]` timestamp compare errors under zsh ("condition expected: >") and loops until timeout without ever matching. The script uses a bash shebang + `[[ > ]]` to avoid this.
- Programmatic Copilot review requests should use GraphQL `requestReviewsByLogin` with `botLogins:["copilot-pull-request-reviewer"]`.
- `gh pr edit --add-reviewer Copilot` and `gh pr edit --add-reviewer @copilot` can return success without a new visible request. Treat them as insufficient unless verification shows a fresh Copilot event, pending reviewer, or review.
- The app slug observed in PR events is `copilot-pull-request-reviewer`.
- A submitted Copilot review may appear as `copilot-pull-request-reviewer[bot]` in PR reviews and as `Copilot` in review comments.
- Keep PRs draft only when the user asks for draft or the local workflow requires it. Otherwise follow the user's requested PR state.
- Copilot code review must be enabled for the repo/org, or these requests succeed silently with no review appearing.
- Include GitHub Copilot review status in the final handoff after creating the PR.
