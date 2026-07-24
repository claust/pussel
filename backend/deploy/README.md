# Backend deployment

The backend runs as a Docker Compose stack on a single server, routed through
a Traefik instance that is already running there for another stack. Every
site-specific value — hostnames, Traefik constraint/entrypoint names, paths,
the server itself — lives in the deploy env file outside the checkout, never
in this repository. The shell snippets below use `$DEPLOY_HOST` (the server)
and `$PUSSEL_ROOT` (the directory holding the checkout) as stand-ins.

## How it deploys

Pull-based, no registry and no secrets in CI:

1. CI (`backend-ci.yml`) builds the image on every PR to catch Dockerfile
   breakage — it never pushes or deploys.
2. On the server, `pussel-deploy.timer` runs [pussel-deploy.sh](pussel-deploy.sh)
   every 5 minutes. When `origin/main` has moved, it hard-resets the checkout,
   rebuilds the image natively, and `docker compose up -d`'s the stack. A
   failed build leaves the previous container running.

Merging to `main` is the deploy button; worst-case latency is ~5 min poll +
the image build.

## Routing and TLS

The existing Traefik owns ports 80/443 and has its Docker provider constrained
to containers that carry its constraint label on the external `gateway`
network. The labels in [docker-compose.yml](docker-compose.yml) opt the backend
into exactly that, so it is published without editing the other stack's compose
file — set `PUSSEL_TRAEFIK_CONSTRAINT` and the two entrypoint names to match
that instance.

Two hostnames are routed (`PUSSEL_HOST_PRIMARY`, `PUSSEL_HOST_ALT`): the
server-native one and the one shipped in the iOS app's Release build. Each is
a proxied CDN CNAME to the server in its own DNS zone; the CDN terminates
public TLS ("Full" SSL mode) and Traefik serves its default certificate
behind it.

## Server layout

```
$PUSSEL_ROOT/
├── repo/          # clone of claust/pussel (managed by pussel-deploy.sh)
└── backend.env    # production secrets + deploy settings, NOT in git
                   # (see backend.env.example)
```

`pussel-deploy.sh` derives the checkout from its own location and the env file
from `$PUSSEL_ROOT/backend.env`; `PUSSEL_REPO_DIR` and `PUSSEL_ENV_FILE`
override both.

## First-time setup / recovery

```bash
ssh $DEPLOY_HOST
mkdir -p $PUSSEL_ROOT
git clone https://github.com/claust/pussel.git $PUSSEL_ROOT/repo
cp $PUSSEL_ROOT/repo/backend/deploy/backend.env.example $PUSSEL_ROOT/backend.env
vi $PUSSEL_ROOT/backend.env   # hostnames, Traefik names, JWT_SECRET, ...

sudo cp $PUSSEL_ROOT/repo/backend/deploy/pussel-deploy.{service,timer} /etc/systemd/system/
sudoedit /etc/systemd/system/pussel-deploy.service   # real User= and ExecStart=
sudo systemctl daemon-reload
sudo systemctl enable --now pussel-deploy.timer

$PUSSEL_ROOT/repo/backend/deploy/pussel-deploy.sh --force   # first build + start
```

## Operations

```bash
# Status / logs
ssh $DEPLOY_HOST 'docker ps --filter name=pussel-backend'
ssh $DEPLOY_HOST 'docker logs -n 100 pussel-backend'
ssh $DEPLOY_HOST 'systemctl list-timers pussel-deploy.timer'
ssh $DEPLOY_HOST 'journalctl -u pussel-deploy.service -n 50'

# Manual redeploy (e.g. after editing backend.env). Double-quoted so the
# local stand-in expands before the command is sent.
ssh $DEPLOY_HOST "$PUSSEL_ROOT/repo/backend/deploy/pussel-deploy.sh --force"

# Health — substitute the PUSSEL_HOST_PRIMARY value from backend.env
curl https://<backend-host>/health
```

Caveats:

- The puzzle store is in-memory, so every deploy drops stored puzzles; the
  iOS app re-uploads with one tap.
- If the server is asleep (wake-on-lan), the backend is unreachable until it
  wakes.
