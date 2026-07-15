#!/usr/bin/env bash
#
# Copilot PR review helper for the gh-pr-copilot-review skill.
#
# Encapsulates the shell-fragile GraphQL + polling steps so they are written and
# tested once, instead of being retyped (and re-broken) every session. Because
# it runs under its own bash shebang it is immune to the caller's interactive
# shell — notably the `[ "$a" \> "$b" ]` string compare that errors under zsh
# with "condition expected: >" and silently defeated the inline watcher.
#
# owner/repo/pr auto-detect from the current repo + checked-out PR branch when
# not passed, so the common case is just:  copilot-review.sh wait
#
# Usage:
#   copilot-review.sh request [--owner O --repo R --pr N]
#   copilot-review.sh verify  [--owner O --repo R --pr N]
#   copilot-review.sh status  [--owner O --repo R --pr N]
#   copilot-review.sh wait    [--owner O --repo R --pr N]
#                             [--baseline TS] [--interval SECS] [--max-iters N]
#
# `wait` prints "COPILOT_REVIEW_READY latest=<ts>" and exits 0 when a Copilot
# review newer than --baseline appears, or "WATCHER_TIMEOUT_NO_NEW_REVIEW" if it
# gives up. For the first round pass no --baseline (any Copilot review matches);
# for re-review rounds pass the previous round's latest Copilot submitted_at so
# it waits for the *new* review. Run it with run_in_background:true to be woken
# on completion instead of foreground-polling.

set -uo pipefail
export LC_ALL=C  # deterministic byte ordering for ISO-8601 timestamp compares

# Copilot shows up as this login in submitted reviews / requested reviewers.
BOT_REVIEW_LOGIN='copilot-pull-request-reviewer[bot]'
BOT_DISPLAY_LOGIN='Copilot'

die() { echo "error: $*" >&2; exit 1; }

command -v gh >/dev/null 2>&1 || die "gh CLI not found on PATH"

cmd="${1:-}"
[ -n "$cmd" ] || die "usage: copilot-review.sh <request|verify|status|wait> [flags]"
shift

OWNER="" REPO="" PR="" BASELINE="" INTERVAL=20 MAX_ITERS=60
while [ $# -gt 0 ]; do
  case "$1" in
    --owner)     OWNER="${2:-}"; shift 2 ;;
    --repo)      REPO="${2:-}"; shift 2 ;;
    --pr)        PR="${2:-}"; shift 2 ;;
    --baseline)  BASELINE="${2:-}"; shift 2 ;;
    --interval)  INTERVAL="${2:-}"; shift 2 ;;
    --max-iters) MAX_ITERS="${2:-}"; shift 2 ;;
    *)           die "unknown argument: $1" ;;
  esac
done

# Auto-detect owner/repo/pr from context when not supplied.
[ -n "$OWNER" ] || OWNER=$(gh repo view --json owner --jq '.owner.login' 2>/dev/null) || true
[ -n "$REPO" ]  || REPO=$(gh repo view --json name --jq '.name' 2>/dev/null) || true
[ -n "$OWNER" ] && [ -n "$REPO" ] || die "could not determine owner/repo — pass --owner and --repo"
[ -n "$PR" ] || PR=$(gh pr view --json number --jq '.number' 2>/dev/null) || true
[ -n "$PR" ] || die "could not determine PR number — pass --pr or checkout the PR branch"

pr_node_id() {
  # shellcheck disable=SC2016  # $owner/$repo/$number are GraphQL vars, not shell — must stay unexpanded
  gh api graphql \
    -f query='query($owner:String!,$repo:String!,$number:Int!){repository(owner:$owner,name:$repo){pullRequest(number:$number){id}}}' \
    -F owner="$OWNER" -F repo="$REPO" -F number="$PR" \
    --jq '.data.repository.pullRequest.id'
}

# Latest submitted_at across Copilot reviews, or empty if none yet.
latest_copilot_ts() {
  gh api "repos/$OWNER/$REPO/pulls/$PR/reviews" --paginate --jq \
    "[.[] | select(.user.login==\"$BOT_REVIEW_LOGIN\" or .user.login==\"$BOT_DISPLAY_LOGIN\")] | max_by(.submitted_at) | .submitted_at // empty" \
    2>/dev/null
}

case "$cmd" in
  request)
    pid=$(pr_node_id)
    [ -n "$pid" ] || die "could not resolve PR node id for $OWNER/$REPO#$PR"
    # shellcheck disable=SC2016  # $pr is a GraphQL var, not shell — must stay unexpanded
    gh api graphql \
      -f query='mutation($pr:ID!){requestReviewsByLogin(input:{pullRequestId:$pr,botLogins:["copilot-pull-request-reviewer"],union:true}){pullRequest{id}}}' \
      -F pr="$pid" >/dev/null \
      && echo "requested Copilot review on $OWNER/$REPO#$PR"
    ;;

  verify)
    echo "== review_requested / copilot_work_started events =="
    gh api "repos/$OWNER/$REPO/issues/$PR/events" --paginate --jq \
      '.[] | select(.event=="review_requested" or .event=="copilot_work_started") | {event, actor:.actor.login, requested:(.requested_reviewer.login // null), app:(.performed_via_github_app.slug // null), created_at}'
    echo "== pending requested reviewers =="
    gh api "repos/$OWNER/$REPO/pulls/$PR/requested_reviewers" --jq '.users[].login' 2>/dev/null || true
    ;;

  status)
    gh api "repos/$OWNER/$REPO/pulls/$PR/reviews" --paginate --jq \
      '.[] | {user:.user.login, state, submitted_at, body:(.body[0:80])}'
    ;;

  wait)
    i=0
    while [ "$i" -lt "$MAX_ITERS" ]; do
      latest=$(latest_copilot_ts)
      if [ -n "$latest" ] && { [ -z "$BASELINE" ] || [[ "$latest" > "$BASELINE" ]]; }; then
        echo "COPILOT_REVIEW_READY latest=$latest"
        exit 0
      fi
      i=$((i + 1))
      sleep "$INTERVAL"
    done
    echo "WATCHER_TIMEOUT_NO_NEW_REVIEW"
    exit 0
    ;;

  *)
    die "unknown command: $cmd (expected request|verify|status|wait)"
    ;;
esac
