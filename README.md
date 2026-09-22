# isummarize-worker

HTTP worker that **summarizes transcripts** with a SKILL (rules) via an OpenAI-compatible Chat Completions API.

**Language:** [English](README.md) · [Русский](README.ru.md)

## What it does

- Input: JSON `{ "text": "<transcript>", "skill": "<summarization rules>" }`.
- Output: a **string** `summary` as returned by the model (plain text or JSON-as-text). The client does not pick the model; `MODEL`, `BASE_URL`, and `API_KEY` live in `.env`.
- Long transcripts are split and summarized with map-reduce **inside one task**. That does not consume extra `WORKERS` slots.
- One Python process: `WORKERS` in `.env` is how many summarization tasks may run at once (not uvicorn `--workers`).

`POST /summarize` returns **202** with a `task_id`. Fetch the result from `/tasks`.

## Requirements

- Python **3.12**
- Virtualenv at `.venv` (use `./.venv/bin/python` and `./.venv/bin/pip` only)
- An OpenAI-compatible endpoint (`BASE_URL` + `API_KEY` + `MODEL`): OpenAI, vLLM, an Ollama proxy, or a corporate gateway
- Disk under `./data` for SQLite, logs, and task tmp (not committed)

## Install and run

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt
```

Create a `.env` in the repo root (see table below). Do not commit it. Then:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Always keep **uvicorn** `--workers 1`. Parallelism of jobs is `WORKERS` in `.env` (slots inside this one process), not extra uvicorn processes.

Check:

```bash
curl -s http://127.0.0.1:8000/health
```

Docker: [Docker Compose](#docker-compose).

## `.env`

Copy names into `.env`. **Do not put real tokens in git or in this README.** Changing a value requires a process restart.

| Variable                  | Meaning                                                                                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `API_TOKEN`               | Bearer key for all routes except `/health` and `/ready`. Empty = nobody is authorized.                                                                           |
| `BASE_URL`                | OpenAI-compatible API base (no secrets in logs).                                                                                                                 |
| `API_KEY`                 | Provider key. Never logged.                                                                                                                                      |
| `MODEL`                   | Chat Completions model name. Not accepted in the request body.                                                                                                   |
| `HOST`                    | Bind address (`127.0.0.1` locally; Docker uses `0.0.0.0`).                                                                                                       |
| `PORT`                    | HTTP port (default `8000`).                                                                                                                                      |
| `DATA_DIR`                | Persistent root (default `./data`): SQLite, logs, and queue tmp at `{DATA_DIR}/tmp/<task_id>/`.                                                                  |
| `SQLITE_PATH`             | Task database (default `./data/tasks.db`). The `summary` column is encrypted at rest (see below). Payload files live under `{DATA_DIR}/tmp/` while queued/running. |
| `LOG_DIR`                 | Application log directory (default `./data/logs`).                                                                                                               |
| `PERFORMANCE_LOG`         | Task metrics CSV (default `./data/logs/performance_log.csv`).                                                                                                    |
| `LOG_ENABLED`             | Application file log + app logger. Default `true`. `false` / `0` / `no` = off. Does not affect CSV / `metric_event`.                                             |
| `LOG_MAX_BYTES`           | Rotate `app.log` when it exceeds this size in bytes. Default `5242880` (5 MiB).                                                                                  |
| `LOG_BACKUP_COUNT`        | How many rotated copies (`app.log.1` … `N`) to keep. Default `5`. Oldest is deleted.                                                                             |
| `PERFORMANCE_LOG_ENABLED` | CSV row + JSON `metric_event` on stdout when a task finishes. Default `true`. `false` / `0` / `no` = off.                                                        |
| `METRICS_ENABLED`         | Application Prometheus metrics on `GET /metrics`. Default `true`. `false` / `0` / `no` = process collectors only; the endpoint stays up.                         |
| `WORKERS`                 | How many **tasks** may run at once in this process. Default `1`. Not uvicorn workers. Required, explicit.                                                        |
| `WORKERS_MAX`             | Alias for `WORKERS` (same value). Exposed as `workers.max` in `GET /health` for [idigest-hub](https://github.com/dude75/idigest-hub) Capacity UI.                 |
| `WORKER_QUEUE_SIZE`       | Max `queued` tasks waiting for a slot. Default `4`. Beyond that: `503` `queue_full`.                                                                             |
| `MAX_PAYLOAD_BYTES`       | Max `POST /summarize` JSON body in bytes. Default `10485760` (10 MiB). Over the limit: HTTP **413** `payload_too_large`.                                          |
| `TASK_TTL_SEC`            | Seconds after `success`/`error` before the SQLite row is deleted. `0` = no TTL (delete only via `DELETE`).                                                       |
| `TASK_MAX_RESTARTS`       | How many times a task found `running` after a process death may be put back in `queued`. Default `1`. After that: `error` with `process_killed`. `0` = fail immediately. |
| `TASK_TIMEOUT_SEC`        | Wall-clock limit for one `running` task. Default `3600`. `0` = no limit (not recommended).                                                                       |
| `LLM_TIMEOUT_SEC`         | Timeout of one HTTP call to the LLM. Default `120`. `0` = no limit (not recommended).                                                                            |
| `LLM_PROBE_TTL_SEC`       | Cache TTL for `GET {BASE_URL}/models` used by `/health` and `/ready`. Default `15`. `0` = probe every time.                                                      |
| `LLM_MAX_RETRIES`         | Extra attempts on connect / 429 / 5xx. Default `2`.                                                                                                              |
| `MAX_TOKENS`              | Optional Chat Completions `max_tokens`. Empty = omit the field.                                                                                                  |
| `TEMPERATURE`             | Chat Completions temperature. Default `0.2`.                                                                                                                     |

Invariants: `WORKERS >= 1`, `TASK_MAX_RESTARTS >= 0`, `TASK_TIMEOUT_SEC >= 0`, `MAX_PAYLOAD_BYTES > 0`, `LOG_MAX_BYTES > 0`, `LOG_BACKUP_COUNT >= 1`, `LLM_PROBE_TTL_SEC >= 0`.

Everything that must survive a restart lives under `./data` (`tasks.db`, logs, **and queue tmp** `{DATA_DIR}/tmp/`). Mount that directory in Docker.

`summary` in `tasks.db` is Fernet-encrypted (AES-128-CBC + HMAC). The key is `SHA-256(API_TOKEN)`, not the raw token. `GET /tasks/{id}` still returns plaintext; the list endpoint never includes the summary. Metadata, `error`, and tmp payload stay unencrypted. This only helps if `tasks.db` leaks without `.env`. Changing `API_TOKEN` makes existing encrypted rows unreadable until TTL or `DELETE`; rows written before this version are still plaintext and keep working.

After a process restart (or `docker compose restart`) unfinished work is restored from SQLite + those tmp files — **not** resumed mid-pipeline:

- `queued` tasks with `text.txt` + `skill.md` on disk are put back on the in-memory queue (FIFO by `timestamp`). `WORKER_QUEUE_SIZE` is **not** applied on restore; new `POST /summarize` still uses the limit.
- A task that was `running` is set back to `queued` and run from scratch if its payload files still exist, at most `TASK_MAX_RESTARTS` times (default `1`). Another process death after that finishes as `error` with `process_killed`. If the files are gone, it finishes as `error` with `interrupted`.
- A `queued` task whose payload files are missing finishes as `error` with `missing_payload` and is not enqueued.
- Graceful shutdown does **not** delete tmp for queued or running tasks. Finished (`success` / `error`) tmp is still cleaned.

## API

All routes except `GET /health` and `GET /ready` require:

`Authorization: Bearer <API_TOKEN>`

Replace `$TOKEN` and `$HOST` in the examples (`http://127.0.0.1:8000`).

### Health (no token)

```bash
curl -s "$HOST/health"
curl -s "$HOST/health" | jq '{version, model, llm, workers}'
```

JSON includes `version` (same as `version.txt`), `model` (from `MODEL` in `.env`), `llm`: `ready` | `unconfigured` | `unavailable`, and `workers`:

```json
{
  "status": "ok",
  "version": "0.1.1",
  "model": "gpt-4o-mini",
  "llm": "ready",
  "workers": {
    "max": 2,
    "active": 0,
    "available": 2
  }
}
```

| Field | Meaning |
| ----- | ------- |
| `workers.max` | Parallel summarize slots on this process (`WORKERS` / `WORKERS_MAX`). |
| `workers.active` | Tasks currently holding a slot (`running`). |
| `workers.available` | Free slots: `max - active`. |

[idigest-hub](https://github.com/dude75/idigest-hub) reads `workers.*` for summarize node Capacity (instead of a legacy 1/1 fallback). No secrets in the response. `unconfigured` means empty `BASE_URL`, `API_KEY`, or `MODEL`. HTTP **200** while the process is up (liveness), even if the provider is down. `ready` means `GET {BASE_URL}/models` returned 2xx; 401/403/429/5xx and network errors are `unavailable`. Probe results are cached for `LLM_PROBE_TTL_SEC`.

### Ready (no token)

```bash
curl -s "$HOST/ready"
```

Same JSON as `/health` (including `model` and `workers`). HTTP **200** only when `llm` is `ready`; otherwise **503**. Point the load balancer / k8s readiness probe here. [idigest-hub](https://github.com/dude75/idigest-hub) uses `/ready` for summarize dispatch; `workers` is informational for Capacity UI only.

### Metrics

```bash
curl -s -H "Authorization: Bearer $TOKEN" "$HOST/metrics"
```

Prometheus text format. Process collectors plus application gauges/counters/histograms (queue, LLM timings, tokens, errors, HTTP). Same Bearer as the rest of the API.

Grafana: import [`grafana/dashboards/isummarize-worker.json`](grafana/dashboards/isummarize-worker.json) (Dashboards → New → Import) and pick the Prometheus that scrapes this endpoint. Example scrape:

```yaml
scrape_configs:
  - job_name: isummarize-worker
    metrics_path: /metrics
    scrape_interval: 15s
    authorization:
      credentials: "<API_TOKEN>"
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

The dashboard covers queue depth / worker slots / running age, queue wait, tasks per minute, HTTP (excluding `/metrics` scrapes), LLM duration, tokens, and errors. CSV `PERFORMANCE_LOG` is separate and is not on this dashboard. Re-import the JSON to pick up dashboard updates (same uid `isummarize-worker`).

### Submit text → 202

```bash
curl -sS -X POST "$HOST/summarize" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"full transcript ...","skill":"summarization rules (SKILL) ..."}'
```

Both `text` and `skill` are required non-empty strings (after trim). The model is **not** accepted in the body.

### Poll one task

```bash
TASK_ID=4f8b9e12-87c2-4911-bca4-d832e12cf900
curl -sS "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN"
```

`status` is `queued` | `running` | `success` | `error`. On success, `summary` is filled. On a **task** error (LLM, payload, pipeline) HTTP is still **200** with `"status": "error"` and an `error` object — keep polling the same URL. Unknown id → **404**.

Example poll loop:

```bash
TASK_ID=$(curl -sS -X POST "$HOST/summarize" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"full transcript ...","skill":"summarization rules"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['meta']['task_id'])")

while true; do
  body=$(curl -sS "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN")
  status=$(printf '%s' "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "$status"
  case "$status" in success|error) printf '%s\n' "$body"; break ;; esac
  sleep 2
done
```

### List tasks

```bash
curl -sS "$HOST/tasks" -H "Authorization: Bearer $TOKEN"
curl -sS "$HOST/tasks?status=success" -H "Authorization: Bearer $TOKEN"
```

Newest first. No `summary` in the list.

### Delete one task

```bash
curl -sS -X DELETE "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN"
```

- `queued` / `success` / `error` → **200**, row removed (queued also drops tmp).
- `running` → **409** `task_running` (in-flight LLM work is not cancelled).

### Purge queue and history

```bash
curl -sS -X DELETE "$HOST/tasks" -H "Authorization: Bearer $TOKEN"
```

Clears the whole queue and finished history. **Does not** cancel a task that is currently `running` (those rows and their tmp stay; HTTP **200**, not **409**). Also removes orphan dirs under `{DATA_DIR}/tmp/` plus leftover CWD `tmp_`*. Does not touch `tasks.db` schema or logs.

JSON **200**:

```json
{
  "status": "ok",
  "purged_queued": 0,
  "purged_finished": 0,
  "purged_tmp": 0,
  "skipped_running": 0
}
```

## Docker Compose

One CPU image (`isummarize-worker:cpu`). The process runs as **uid/gid 1001** (not root). Compose mounts `./data:/data` so SQLite, logs, and tmp survive `docker compose restart`.

1. Copy `.env.example` → `.env` and fill `API_TOKEN`, `BASE_URL`, `API_KEY`, `MODEL`.
2. Create `./data` if it does not exist. On Linux it must be writable by uid 1001: `sudo chown -R 1001:1001 data`.

```bash
docker compose up --build
```

Add `-d` to run in the background (`docker compose logs -f` for logs). Published port: `8000:8000`. Then the same API `curl` examples against `http://127.0.0.1:8000`.

```bash
docker compose down
```

`./data` on the host is not deleted.

## Typical errors

| What you see                                                      | Meaning                                                                                                      |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| HTTP **401**, `error.code = unauthorized`                         | Missing/wrong `Authorization: Bearer …`, or empty `API_TOKEN`.                                               |
| HTTP **503**, `error.code = llm_unconfigured`                     | `POST /summarize` while `BASE_URL` / `API_KEY` / `MODEL` is empty.                                           |
| HTTP **503**, `error.code = llm_unavailable`                      | `POST /summarize` while the LLM probe is not 2xx (cached for `LLM_PROBE_TTL_SEC`).                           |
| HTTP **503**, `error.code = queue_full`                           | Too many `queued` tasks (`WORKER_QUEUE_SIZE`). Wait or raise the limit and restart.                          |
| HTTP **413**, `error.code = payload_too_large`                    | `POST /summarize` body larger than `MAX_PAYLOAD_BYTES`.                                                      |
| HTTP **422**                                                      | Empty `text` / `skill` (after trim), or extra fields such as `model`.                                        |
| HTTP **200**, `status=error`, `error.code = missing_payload`      | `text.txt` / `skill.md` for a queued/restored task is gone from `{DATA_DIR}/tmp/`.                           |
| HTTP **200**, `status=error`, `error.code = interrupted`          | Process died while the task was `running` and the payload files were missing after restart.                  |
| HTTP **200**, `status=error`, `error.code = process_killed`       | Process died while `running` more times than `TASK_MAX_RESTARTS`.                                            |
| HTTP **200**, `status=error`, `error.code = llm_unconfigured`     | Empty `BASE_URL` / `API_KEY` / `MODEL`.                                                                      |
| HTTP **200**, `status=error`, `error.code = llm_unavailable`      | Network, provider 401/403, or 5xx after retries.                                                             |
| HTTP **200**, `status=error`, `error.code = llm_timeout`          | HTTP call to the LLM exceeded `LLM_TIMEOUT_SEC`.                                                             |
| HTTP **200**, `status=error`, `error.code = task_timeout`         | The running task exceeded `TASK_TIMEOUT_SEC`.                                                                |
| HTTP **200**, `status=error`, `error.code = llm_bad_response`     | Empty or unusable `choices[0].message.content`.                                                              |
| HTTP **200**, `status=error`, `error.code = text_too_long`        | Even one chunk plus the SKILL does not fit the character budget.                                             |
| HTTP **200**, `status=error`, `error.code = pipeline_error`       | Any other pipeline failure.                                                                                  |
