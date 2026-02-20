# Semantics Cluster

Pool your home GPUs across multiple machines into a single AI/ML cluster, connected over Tailscale.  
Ships with an audio transcription API as a working example.

---

## What Is This?

You have GPUs spread across different machines at home. This project wires them together into one [Ray](https://www.ray.io/) cluster using [Tailscale](https://tailscale.com) — a free, zero-config mesh VPN that works through NAT with no port forwarding.

The same `docker-compose.yml` runs on every machine. You pick the profile:
- **Head machine** → runs the Ray head, API, storage, and monitoring.
- **Worker machines** → join the cluster and contribute their GPUs.

As an example workload, a .NET REST API accepts audio files, distributes transcription across all GPUs using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and returns results. You can add your own Ray jobs too.

```
User ──POST /transcribe──▸ API (.NET) ──▸ Upload to Azurite (blob storage)
                                     └──▸ Submit Ray Job
                                              │
                                     ┌────────┼────────┐
                                     ▼        ▼        ▼
                                  GPU 0    GPU 1    GPU 2  ...
                                (machine A) (machine A) (machine B)
                                     │        │        │
                                     └────────┼────────┘
                                              ▼
                                    Merge + upload result JSON
```

---

## Prerequisites

On **every machine** (head and workers):

- **Docker** + **Docker Compose** (v2)
- **NVIDIA GPU(s)** with drivers installed — Docker handles the rest via the GPU-enabled Ray image
- A **free Tailscale account** (see below)

---

## Set Up Tailscale (One Time)

1. Go to [tailscale.com](https://tailscale.com) and **sign up** — it's free for personal use (up to 100 devices).
2. Once logged in, go to **Settings → Keys** → click **Generate auth key**.
3. Check **Reusable** so all your machines can use the same key.  
   *(Optional: check **Ephemeral** if you want nodes to auto-deregister when containers stop.)*
4. **Copy the key** — you'll paste it into your `.env` file next.

> The key looks like: `tskey-auth-kABC123DEF456-rest...`

---

## Quick Start — Head Machine

This is the machine that runs the API, Ray scheduler, storage, and monitoring. It also contributes its own GPU(s) to the cluster.

```bash
# 1. Clone the repo
git clone https://github.com/famda/semantics-cluster.git
cd semantics-cluster

# 2. Create your .env file from the example
cp .env.example .env

# 3. Edit .env — fill in these two:
#    TS_AUTHKEY=tskey-auth-your-key-here
#    GF_ADMIN_PASSWORD=pick-something
#    (leave AZURITE_CONN_STR as-is — the default works)

# 4. Start the head
docker compose --profile head up -d --build
```

That's it. Once it's running you have:

| Service | URL |
|---|---|
| Transcription API | `http://localhost:5080` |
| Swagger UI (dev) | `http://localhost:5080/swagger` |
| Ray Dashboard | `http://localhost:8265` |
| Grafana | `http://localhost:3000` |
| Prometheus | `http://localhost:9090` |

![Ray Dashboard — Overview](docs/ray-dashboard-overview.png)

---

## Adding Workers

Repeat this on **each additional machine** that has a GPU you want to add to the cluster. You can add as many as you want.

```bash
# 1. Clone the repo
git clone https://github.com/famda/semantics-cluster.git
cd semantics-cluster

# 2. Create your .env file
cp .env.example .env

# 3. Edit .env — fill in:
#    TS_AUTHKEY=tskey-auth-your-key-here      ← same key as the head
#    TS_HOSTNAME=ray-worker-1                  ← unique name for this machine
#    RAY_HEAD_ADDRESS=ray-head:6379            ← leave as-is (default works)

# 4. Start the worker
docker compose --profile node up -d --build
```

Open the Ray Dashboard on the head machine (`http://localhost:8265`) — the worker should appear in the cluster:

![Ray Dashboard — Cluster nodes](docs/ray-dashboard-cluster.png)

For additional machines, just change `TS_HOSTNAME` each time:
- Machine 2: `TS_HOSTNAME=ray-worker-2`
- Machine 3: `TS_HOSTNAME=ray-worker-3`
- ...and so on.

---

## Try It — Transcribe Something

Open the Swagger UI at **`http://localhost:5080/swagger`**, expand `POST /transcribe`, click **Try it out**, upload a file, and hit **Execute**.

![Swagger UI — POST /transcribe](docs/api-swagger.png)

You'll get back a `202 Accepted` response with:

```json
{
  "jobId": "c0cf48703f634f18b23936f87b972f3f",
  "raySubmissionId": "transcription-c0cf48703f634f18b23936f87b972f3f",
  "status": "PENDING",
  "blobPath": "files/c0cf.../my-audio.mp3",
  "resultPath": "files/c0cf.../transcription.json"
}
```

Track progress in the Ray Dashboard → **Jobs** tab (`http://localhost:8265/#/jobs`). When the job finishes, the transcription JSON is available in Azurite at the `resultPath`.

![Ray Dashboard — Jobs](docs/ray-dashboard-jobs.png)

---

## How It Works

1. **API** receives the file upload, stores it in Azurite (local blob storage emulator), and submits a Ray job.
2. **Ray job** downloads the file, decodes it with ffmpeg, and chunks the audio into ~150-second segments.
3. **Chunks are distributed** across all available GPUs in the cluster — one [faster-whisper](https://github.com/SYSTRAN/faster-whisper) actor per GPU.
4. **Segments are transcribed** in parallel, then merged, deduplicated, and the result JSON is uploaded back to Azurite.
5. **Model:** `distil-large-v3.5` by default.

---

## Monitoring

| Tool | URL | Notes |
|---|---|---|
| **Grafana** | `http://localhost:3000` | Login: `admin` / your `GF_ADMIN_PASSWORD`. Dashboards are pre-provisioned. |
| **Prometheus** | `http://localhost:9090` | Auto-discovers all Ray nodes. |
| **Ray Dashboard** | `http://localhost:8265` | Cluster status, jobs, actors, logs. |

![Ray Dashboard — Metrics](docs/ray-dashboard-metrics.png)

Scroll down in Metrics to see per-node GPU utilization and VRAM usage across the whole cluster:

![Ray Dashboard — GPU Metrics](docs/ray-dashboard-gpu-metrics.png)

---

## Environment Variables

All config lives in the `.env` file next to `docker-compose.yml`.  
Copy from `.env.example` and fill in the values.

| Variable | Where | Description |
|---|---|---|
| `TS_AUTHKEY` | **All machines** | Your Tailscale auth key (reusable). |
| `TS_HOSTNAME` | **Workers only** | Unique name per worker (`ray-worker-1`, `ray-worker-2`, …). |
| `RAY_HEAD_ADDRESS` | **Workers only** | Head node address. Default `ray-head:6379` works out of the box. |
| `GF_ADMIN_PASSWORD` | **Head only** | Grafana admin password. Set before first run. |
| `AZURITE_CONN_STR` | **Head only** | Azurite connection string. Default in `.env.example` works as-is. |

---

## Ports Reference

All exposed on the **head machine** only:

| Port | Service |
|---|---|
| `5080` | Transcription API |
| `6379` | Ray GCS (cluster communication) |
| `8265` | Ray Dashboard |
| `9090` | Prometheus |
| `3000` | Grafana |
| `10000` | Azurite Blob Storage |

---

## License

[MIT](LICENSE)