# Backend deployment (delectosoft home server)

The backend runs as a Docker Compose stack on the home server, routed through
the Appwrite stack's existing Traefik at **https://pussel.sabeltiger.dk**.

## How it deploys

Pull-based, no registry and no secrets in CI:

1. CI (`backend-ci.yml`) builds the image on every PR to catch Dockerfile
   breakage — it never pushes or deploys.
2. On the server, `pussel-deploy.timer` runs [pussel-deploy.sh](pussel-deploy.sh)
   every 5 minutes. When `origin/main` has moved, it hard-resets the checkout
   at `/home/claus/pussel/repo`, rebuilds the image natively, and
   `docker compose up -d`'s the stack. A failed build leaves the previous
   container running.

Merging to `main` is the deploy button; worst-case latency is ~5 min poll +
the image build.

## Routing and TLS

Appwrite's Traefik (`appwrite-traefik`, owns ports 80/443) has its Docker
provider constrained to containers labeled
`traefik.constraint-label-stack=appwrite` on the external `gateway` network.
The labels in [docker-compose.yml](docker-compose.yml) register the backend
under two hostnames: `pussel.sabeltiger.dk` (home-server-native) and
`pussel.thomasen.dk` (the URL shipped in the iOS app's Release build,
formerly pointing at Azure). Each is a proxied Cloudflare CNAME →
`sabeltiger.dk` in its own zone; Cloudflare terminates public TLS (SSL mode
"Full"), Traefik serves its default certificate to Cloudflare — the same
recipe as `photos.sabeltiger.dk` in the home-server repo.

## Server layout

```
/home/claus/pussel/
├── repo/          # clone of claust/pussel (managed by pussel-deploy.sh)
└── backend.env    # production secrets, NOT in git (see backend.env.example)
```

## First-time setup / recovery

```bash
ssh claus@delectosoft
mkdir -p /home/claus/pussel
git clone https://github.com/claust/pussel.git /home/claus/pussel/repo
cp /home/claus/pussel/repo/backend/deploy/backend.env.example /home/claus/pussel/backend.env
vi /home/claus/pussel/backend.env   # JWT_SECRET, GOOGLE_CLIENT_ID, ...

sudo cp /home/claus/pussel/repo/backend/deploy/pussel-deploy.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pussel-deploy.timer

/home/claus/pussel/repo/backend/deploy/pussel-deploy.sh --force   # first build + start
```

## Operations

```bash
# Status / logs
ssh claus@delectosoft 'docker ps --filter name=pussel-backend'
ssh claus@delectosoft 'docker logs -n 100 pussel-backend'
ssh claus@delectosoft 'systemctl list-timers pussel-deploy.timer'
ssh claus@delectosoft 'journalctl -u pussel-deploy.service -n 50'

# Manual redeploy (e.g. after editing backend.env)
ssh claus@delectosoft '/home/claus/pussel/repo/backend/deploy/pussel-deploy.sh --force'

# Health
curl https://pussel.sabeltiger.dk/health
```

Caveats:

- The puzzle store is in-memory, so every deploy drops stored puzzles; the
  iOS app re-uploads with one tap.
- If the server is asleep (it has wake-on-lan, see the home-server repo),
  the backend is unreachable until it wakes.
