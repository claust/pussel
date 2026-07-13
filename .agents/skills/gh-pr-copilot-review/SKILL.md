---
name: gh-pr-copilot-review
description: Commit local changes, push a branch, create a GitHub pull request, request GitHub Copilot as reviewer, verify Copilot actually started through PR events, and monitor Copilot review progress. Use when the user asks Codex to publish changes to GitHub, open a PR, request Copilot review, check Copilot review progress, or set up periodic PR review monitoring.
---

# GitHub PR Copilot Review

## Overview

Publish local work to GitHub end to end: inspect scope, branch, stage, commit, push, create a PR, request GitHub Copilot code review, verify the request through GitHub issue events, monitor Copilot review output, address substantive feedback, push fixes, re-request Copilot review, and continue until unresolved feedback is only nits or non-actionable.

Codex does not receive passive GitHub callbacks in an ordinary thread. For follow-up progress, actively poll GitHub or create a Codex heartbeat/automation when the user asks to keep watching.

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

   - If on a detached HEAD, `main`, `master`, or the default branch, create a `codex/<short-description>` branch.
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

8. Request Copilot review using the GraphQL bot-review path below. Use this for both initial review requests and re-review requests after pushing fixes. Do not rely on `gh pr edit --add-reviewer Copilot` or `gh pr edit --add-reviewer @copilot`; those commands can return success without creating a visible pending Copilot request or a new `copilot_work_started` event.

## Request Or Re-Request Copilot Review

Use GraphQL `requestReviewsByLogin` with `botLogins:["copilot-pull-request-reviewer"]`. This is the reliable programmatic equivalent of the GitHub web UI's Copilot review/re-review request.

```bash
PR_NUMBER=<pr-number>
OWNER=<owner>
REPO=<repo>

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

Verify Copilot through issue events and requested reviewers. GitHub may leave `reviewRequests` empty in `gh pr view`, and submitted Copilot reviews remove Copilot from the pending requested-reviewer list.

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

7. If the update or conflict resolution changes code that Copilot previously reviewed, re-request Copilot review with the GraphQL bot-review path.

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

6. Re-request Copilot review with the GraphQL bot-review path above after every push that addresses Copilot feedback.

7. Keep the branch updated with the base branch while this loop is running. If the base branch changes, merge it into the PR branch, resolve any conflicts, push, and wait for CI to rerun.

8. Repeat monitoring, updating from base, fixing, pushing, replying, resolving, and re-requesting until:

   - all CI checks pass,
   - merge state is clean or mergeable,
   - the branch is not behind the base branch,
   - there are no unresolved actionable threads,
   - any remaining unresolved comments are clearly nits or non-actionable,
   - Copilot has either submitted a post-fix review or the latest re-request is visibly pending/started.

## Periodic Follow-Up

If the user asks to watch, monitor, check back, notify them, or get callbacks, create a Codex automation rather than claiming passive callbacks exist.

- Prefer a heartbeat automation attached to the current thread for short-term review monitoring.
- Use a cron automation for detached repository monitoring.
- The automation prompt should ask Codex to inspect the PR using the GitHub CLI/API commands above and report only meaningful changes: new Copilot review, new comments, failed checks, or merge readiness.

## Important Details

- Programmatic Copilot review requests should use GraphQL `requestReviewsByLogin` with `botLogins:["copilot-pull-request-reviewer"]`.
- `gh pr edit --add-reviewer Copilot` and `gh pr edit --add-reviewer @copilot` can return success without a new visible request. Treat them as insufficient unless verification shows a fresh Copilot event, pending reviewer, or review.
- The app slug observed in PR events is `copilot-pull-request-reviewer`.
- A submitted Copilot review may appear as `copilot-pull-request-reviewer[bot]` in PR reviews and as `Copilot` in review comments.
- Keep PRs draft only when the user asks for draft or the local workflow requires it. Otherwise follow the user's requested PR state.
- Include GitHub Copilot review status in the final handoff after creating the PR.
