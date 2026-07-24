#!/usr/bin/env bash
# Pull-based deploy of the pussel backend on the server.
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

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${PUSSEL_REPO_DIR:-$(cd -- "$script_dir/../.." && pwd)}
BRANCH=${PUSSEL_BRANCH:-main}
COMPOSE_FILE="$REPO_DIR/backend/deploy/docker-compose.yml"

# Secrets and site-specific deploy settings live next to the checkout, not in
# it. Compose substitutes the PUSSEL_* settings (hostnames, Traefik names)
# from the environment, so export those; the application secrets stay in the
# file and reach the container through the compose `env_file`.
PUSSEL_ENV_FILE=${PUSSEL_ENV_FILE:-$(dirname -- "$REPO_DIR")/backend.env}
export PUSSEL_ENV_FILE
while IFS='=' read -r key value; do
    export "$key=$value"
done < <(grep -E '^PUSSEL_[A-Za-z0-9_]+=' "$PUSSEL_ENV_FILE")

cd "$REPO_DIR"
git fetch --quiet origin "$BRANCH"

local_rev=$(git rev-parse HEAD)
remote_rev=$(git rev-parse "origin/$BRANCH")

if [[ "$local_rev" == "$remote_rev" && "${1:-}" != "--force" ]]; then
    exit 0
fi

echo "Deploying $BRANCH: ${local_rev:0:9} -> ${remote_rev:0:9}"
# The checkout is deploy-managed, never hand-edited: force-sync straight to
# origin (works even if the local branch is missing) and drop any local
# drift — a stray chmod once wedged a plain `git checkout` here. Secrets
# live outside the checkout, so cleaning untracked files is safe.
git checkout --quiet --force -B "$BRANCH" "origin/$BRANCH"
git clean -fdq

docker compose -f "$COMPOSE_FILE" build backend
docker compose -f "$COMPOSE_FILE" up -d backend

# Drop superseded image layers so repeated deploys don't fill the disk.
# Label-filtered so only this compose project's dangling images are pruned —
# the server runs other stacks whose layers must be left alone.
docker image prune -f --filter "label=com.docker.compose.project=pussel" >/dev/null

echo "Deployed $(git rev-parse --short HEAD)"
