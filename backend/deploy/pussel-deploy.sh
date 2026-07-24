#!/usr/bin/env bash
# Pull-based deploy of the pussel backend on delectosoft.
#
# Run periodically by pussel-deploy.timer (see the systemd units next to this
# script): when origin/main has moved, hard-reset the checkout to it, rebuild
# the image, and restart the stack. A failed build leaves the running
# container untouched (set -e aborts before `up`).
#
# Usage: pussel-deploy.sh [--force]
#   --force  redeploy even when the checkout is already at origin/main
#            (first-time bring-up, env changes, manual rebuilds).

set -euo pipefail

REPO_DIR=${PUSSEL_REPO_DIR:-/home/claus/pussel/repo}
BRANCH=${PUSSEL_BRANCH:-main}
COMPOSE_FILE="$REPO_DIR/backend/deploy/docker-compose.yml"

cd "$REPO_DIR"
git fetch --quiet origin "$BRANCH"

local_rev=$(git rev-parse HEAD)
remote_rev=$(git rev-parse "origin/$BRANCH")

if [[ "$local_rev" == "$remote_rev" && "${1:-}" != "--force" ]]; then
    exit 0
fi

echo "Deploying $BRANCH: ${local_rev:0:9} -> ${remote_rev:0:9}"
# --force: the checkout is deploy-managed, never hand-edited; any local
# drift (e.g. a stray chmod) must not wedge the deploy loop.
git checkout --quiet --force "$BRANCH"
git reset --hard --quiet "origin/$BRANCH"

docker compose -f "$COMPOSE_FILE" build backend
docker compose -f "$COMPOSE_FILE" up -d backend

# Drop superseded image layers so repeated deploys don't fill the disk.
docker image prune -f >/dev/null

echo "Deployed $(git rev-parse --short HEAD)"
