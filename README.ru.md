# isummarize-worker

HTTP-воркер **саммаризации транскриптов** по правилам (SKILL) через OpenAI-совместимый Chat Completions API.

**Язык:** [English](README.md) · [Русский](README.ru.md)

## Что это

- На вход: JSON `{ "text": "<транскрипт>", "skill": "<правила саммаризации>" }`.
- На выход: строка `summary` как вернула модель (текст или JSON-как-текст). Модель клиент не выбирает: `MODEL`, `BASE_URL` и `API_KEY` живут в `.env`.
- Длинный транскрипт режется и саммарится map-reduce **внутри одной задачи**. Это не занимает лишние слоты `WORKERS`.
- Один процесс Python: `WORKERS` в `.env` — сколько задач саммаризации могут идти одновременно (не uvicorn `--workers`).

`POST /summarize` отвечает **202** и `task_id`. Результат забирается через `/tasks`.

## Требования

- Python **3.12**
- Виртуальное окружение `.venv` (только `./.venv/bin/python` и `./.venv/bin/pip`)
- OpenAI-совместимый endpoint (`BASE_URL` + `API_KEY` + `MODEL`): OpenAI, vLLM, Ollama-прокси или корпоративный gateway
- Диск под `./data` для SQLite, логов и tmp очереди задач (в git не коммитится)

## Установка и запуск

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt
```

Создайте `.env` в корне репозитория (таблица ниже). Файл не коммитить. Затем:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Всегда **uvicorn** `--workers 1`. Параллелизм задач — это `WORKERS` в `.env` (слоты внутри этого процесса), а не дополнительные процессы uvicorn.

Проверка:

```bash
curl -s http://127.0.0.1:8000/health
```

Docker: [Docker Compose](#docker-compose).

## `.env`

Имена переменных — в `.env`. **Реальные токены не класть в git и не копировать в README.** Смена значения требует перезапуска процесса.

| Переменная                | Смысл                                                                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `API_TOKEN`               | Bearer-ключ для всех маршрутов, кроме `/health` и `/ready`. Пустой = никто не пройдёт.                                                    |
| `BASE_URL`                | Базовый URL OpenAI-совместимого API (без секретов в логах).                                                                               |
| `API_KEY`                 | Ключ провайдера. Никогда не логируется.                                                                                                   |
| `MODEL`                   | Имя модели Chat Completions. В теле запроса не принимается.                                                                               |
| `HOST`                    | Интерфейс (`127.0.0.1` локально; в Docker — `0.0.0.0`).                                                                                   |
| `PORT`                    | HTTP-порт (по умолчанию `8000`).                                                                                                          |
| `DATA_DIR`                | Корень персистентных данных (по умолчанию `./data`): SQLite, логи и tmp очереди `{DATA_DIR}/tmp/<task_id>/`.                              |
| `SQLITE_PATH`             | БД задач (по умолчанию `./data/tasks.db`). Колонка `summary` шифруется at rest (см. ниже). Payload лежит в `{DATA_DIR}/tmp/` пока queued/running. |
| `LOG_DIR`                 | Каталог прикладных логов (по умолчанию `./data/logs`).                                                                                    |
| `PERFORMANCE_LOG`         | CSV метрик задач (по умолчанию `./data/logs/performance_log.csv`).                                                                        |
| `LOG_ENABLED`             | Прикладной лог-файл + app-logger. По умолчанию `true`. `false` / `0` / `no` — выкл.                                                       |
| `LOG_MAX_BYTES`           | Ротация `app.log` при превышении размера в байтах. По умолчанию `5242880` (5 МиБ).                                                        |
| `LOG_BACKUP_COUNT`        | Сколько копий (`app.log.1` … `N`) хранить. По умолчанию `5`. Самая старая удаляется.                                                      |
| `PERFORMANCE_LOG_ENABLED` | Строка CSV + JSON `metric_event` в stdout при завершении задачи. По умолчанию `true`.                                                     |
| `METRICS_ENABLED`         | Прикладные метрики Prometheus на `GET /metrics`. По умолчанию `true`. `false` / `0` / `no` — только process collectors.                    |
| `WORKERS`                 | Сколько **задач** могут выполняться одновременно в этом процессе. По умолчанию `1`. Не uvicorn workers. Обязательная явная настройка.     |
| `WORKERS_MAX`             | Синоним `WORKERS` (то же значение). В `GET /health` → `workers.max` для Capacity UI [idigest-hub](https://github.com/dude75/idigest-hub). |
| `WORKER_QUEUE_SIZE`       | Максимум задач в `queued`. По умолчанию `4`. Сверх лимита: `503` `queue_full`.                                                             |
| `MAX_PAYLOAD_BYTES`       | Максимум JSON-тела `POST /summarize` в байтах. По умолчанию `10485760` (10 МиБ). Сверх лимита: HTTP **413** `payload_too_large`.           |
| `TASK_TTL_SEC`            | Секунд после `success`/`error` до удаления строки SQLite. `0` — без TTL.                                                                  |
| `TASK_MAX_RESTARTS`       | Сколько раз вернуть `running` в `queued` после смерти процесса. По умолчанию `1`. Дальше `error` `process_killed`.                        |
| `TASK_TIMEOUT_SEC`        | Лимит wall-clock на одну `running`-задачу. По умолчанию `3600`. `0` — без лимита (не рекомендуется).                                      |
| `LLM_TIMEOUT_SEC`         | Таймаут одного HTTP-вызова к LLM. По умолчанию `120`. `0` — без лимита (не рекомендуется).                                                |
| `LLM_PROBE_TTL_SEC`       | TTL кэша `GET {BASE_URL}/models` для `/health` и `/ready`. По умолчанию `15`. `0` — проба каждый раз.                                     |
| `LLM_MAX_RETRIES`         | Повторы при сети / 429 / 5xx. По умолчанию `2`.                                                                                           |
| `MAX_TOKENS`              | Опциональное поле `max_tokens`. Пустое = не отправлять.                                                                                   |
| `TEMPERATURE`             | Temperature Chat Completions. По умолчанию `0.2`.                                                                                         |

Инварианты: `WORKERS >= 1`, `TASK_MAX_RESTARTS >= 0`, `TASK_TIMEOUT_SEC >= 0`, `MAX_PAYLOAD_BYTES > 0`, `LOG_MAX_BYTES > 0`, `LOG_BACKUP_COUNT >= 1`, `LLM_PROBE_TTL_SEC >= 0`.

Всё, что должно пережить рестарт, живёт в `./data` (`tasks.db`, логи и **tmp очереди** `{DATA_DIR}/tmp/`). Этот каталог монтируется в Docker.

Колонка `summary` в `tasks.db` хранится в Fernet (AES-128-CBC + HMAC). Ключ — `SHA-256(API_TOKEN)`, не сырой токен. `GET /tasks/{id}` по-прежнему отдаёт открытый текст; в списке задач саммари нет. Метаданные, `error` и tmp payload не шифруются. Это защита только от утечки `tasks.db` без `.env`. Смена `API_TOKEN` делает уже зашифрованные строки нечитаемыми до TTL или `DELETE`; строки, записанные до этой версии, остаются обычным текстом и читаются как раньше.

После рестарта процесса (или `docker compose restart`) незаконченная работа поднимается из SQLite + этих tmp-файлов — **не** продолжается с середины пайплайна:

- `queued` с файлами `text.txt` + `skill.md` возвращаются в in-memory очередь (FIFO по `timestamp`). `WORKER_QUEUE_SIZE` на restore **не** применяется; новые `POST /summarize` по-прежнему режутся лимитом.
- Задача, которая была `running`, сбрасывается в `queued` и гоняется с начала, если payload на диске есть, не больше `TASK_MAX_RESTARTS` раз (по умолчанию `1`). Иначе `error` `process_killed`. Если файлов нет — `error` `interrupted`.
- `queued` без файлов — `error` `missing_payload`, в очередь не ставится.
- Graceful shutdown **не** удаляет tmp у queued/running. У `success` / `error` tmp чистится.

## API

Все маршруты кроме `GET /health` и `GET /ready` требуют:

`Authorization: Bearer <API_TOKEN>`

В примерах замените `$TOKEN` и `$HOST` (`http://127.0.0.1:8000`).

### Health (без токена)

```bash
curl -s "$HOST/health"
curl -s "$HOST/health" | jq '{version, model, llm, workers}'
```

В JSON: `version` (как в `version.txt`), `model` (из `MODEL` в `.env`), `llm`: `ready` | `unconfigured` | `unavailable`, и блок `workers`:

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

| Поле | Смысл |
| ---- | ----- |
| `workers.max` | Параллельных слотов саммаризации на этом процессе (`WORKERS` / `WORKERS_MAX`). |
| `workers.active` | Задач, занимающих слот (`running`). |
| `workers.available` | Свободных слотов: `max - active`. |

[idigest-hub](https://github.com/dude75/idigest-hub) читает `workers.*` для Capacity summarize-нод (вместо legacy fallback 1/1). Секреты не светятся. `unconfigured` — пустые `BASE_URL`, `API_KEY` или `MODEL`. HTTP **200**, пока процесс жив (liveness), даже если провайдер лежит. `ready` — `GET {BASE_URL}/models` ответил 2xx; 401/403/429/5xx и сеть — `unavailable`. Результат пробы кэшируется на `LLM_PROBE_TTL_SEC`.

### Ready (без токена)

```bash
curl -s "$HOST/ready"
```

Тот же JSON, что у `/health` (включая `model` и `workers`). HTTP **200** только если `llm` = `ready`, иначе **503**. Сюда смотрит балансер / k8s readiness. [idigest-hub](https://github.com/dude75/idigest-hub) для dispatch summarize смотрит `/ready`; `workers` — только для Capacity в UI.

### Метрики

```bash
curl -s -H "Authorization: Bearer $TOKEN" "$HOST/metrics"
```

Prometheus text. Process collectors плюс прикладные (очередь, длительность LLM, токены, ошибки, HTTP). Тот же Bearer, что и у остального API.

Grafana: импорт [`grafana/dashboards/isummarize-worker.json`](grafana/dashboards/isummarize-worker.json). Пример scrape:

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

Дашборд: глубина очереди / слоты воркеров / возраст running, ожидание в очереди, задачи в минуту, HTTP (без scrape `/metrics`), длительность LLM, токены, ошибки. CSV `PERFORMANCE_LOG` на дашборд не выводится. Чтобы подтянуть обновление дашборда — импортируйте JSON ещё раз (тот же uid `isummarize-worker`).

### Отправить текст → 202

```bash
curl -sS -X POST "$HOST/summarize" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"полный транскрипт ...","skill":"правила саммаризации (SKILL) ..."}'
```

Оба поля обязательные непустые строки после trim. Модель в теле **не** принимается.

### Поллинг одной задачи

```bash
TASK_ID=4f8b9e12-87c2-4911-bca4-d832e12cf900
curl -sS "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN"
```

`status`: `queued` | `running` | `success` | `error`. На success заполнено `summary`. Ошибка пайплайна — HTTP **200**, `"status": "error"` и объект `error`. Неизвестный id → **404**.

Пример цикла поллинга:

```bash
TASK_ID=$(curl -sS -X POST "$HOST/summarize" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"полный транскрипт ...","skill":"правила саммаризации"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['meta']['task_id'])")

while true; do
  body=$(curl -sS "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN")
  status=$(printf '%s' "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "$status"
  case "$status" in success|error) printf '%s\n' "$body"; break ;; esac
  sleep 2
done
```

### Список задач

```bash
curl -sS "$HOST/tasks" -H "Authorization: Bearer $TOKEN"
curl -sS "$HOST/tasks?status=success" -H "Authorization: Bearer $TOKEN"
```

Newest first. Без `summary`.

### Удалить одну задачу

```bash
curl -sS -X DELETE "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN"
```

- `queued` / `success` / `error` → **200**, строка снята (у queued ещё и tmp).
- `running` → **409** `task_running` (идущий вызов LLM не отменяется).

### Purge очереди и истории

```bash
curl -sS -X DELETE "$HOST/tasks" -H "Authorization: Bearer $TOKEN"
```

Чистит очередь и историю. **Не** отменяет текущий `running` (строки и tmp остаются; HTTP **200**, не **409**). Ещё снимает orphan-каталоги в `{DATA_DIR}/tmp/` и хвосты CWD `tmp_`*.

## Docker Compose

Один CPU-образ (`isummarize-worker:cpu`). Процесс идёт от **uid/gid 1001** (не root). Compose монтирует `./data:/data`, чтобы SQLite, логи и tmp переживали `docker compose restart`.

1. Скопируйте `.env.example` → `.env` и заполните `API_TOKEN`, `BASE_URL`, `API_KEY`, `MODEL`.
2. Создайте `./data`, если его нет. На Linux каталог должен быть доступен на запись uid 1001: `sudo chown -R 1001:1001 data`.

```bash
docker compose up --build
```

Фон: `-d`, логи: `docker compose logs -f`. Порт: `8000:8000`. Дальше те же `curl` к `http://127.0.0.1:8000`.

```bash
docker compose down
```

`./data` на хосте не удаляется.

## Типичные ошибки

| Что видно                                                           | Смысл                                                                                 |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| HTTP **401**, `error.code = unauthorized`                           | Нет/неверный Bearer или пустой `API_TOKEN`.                                           |
| HTTP **503**, `error.code = llm_unconfigured`                       | `POST /summarize` при пустых `BASE_URL` / `API_KEY` / `MODEL`.                        |
| HTTP **503**, `error.code = llm_unavailable`                        | `POST /summarize`, пока проба LLM не 2xx (кэш `LLM_PROBE_TTL_SEC`).                   |
| HTTP **503**, `error.code = queue_full`                             | Слишком много `queued` (`WORKER_QUEUE_SIZE`).                                         |
| HTTP **413**, `error.code = payload_too_large`                      | Тело `POST /summarize` больше `MAX_PAYLOAD_BYTES`.                                    |
| HTTP **422**                                                        | Пустые `text` / `skill` после trim или лишние поля вроде `model`.                     |
| HTTP **200**, `status=error`, `error.code = missing_payload`        | Нет `text.txt` / `skill.md` после restore.                                            |
| HTTP **200**, `status=error`, `error.code = interrupted`            | Процесс умер на `running` и payload пропал.                                           |
| HTTP **200**, `status=error`, `error.code = process_killed`         | Слишком много рестартов `running`.                                                    |
| HTTP **200**, `status=error`, `error.code = llm_unconfigured`       | Пустые `BASE_URL` / `API_KEY` / `MODEL`.                                              |
| HTTP **200**, `status=error`, `error.code = llm_unavailable`        | Сеть, 401/403 провайдера или 5xx после ретраев.                                       |
| HTTP **200**, `status=error`, `error.code = llm_timeout`            | Таймаут HTTP к LLM.                                                                   |
| HTTP **200**, `status=error`, `error.code = task_timeout`           | Задача превысила `TASK_TIMEOUT_SEC`.                                                  |
| HTTP **200**, `status=error`, `error.code = llm_bad_response`       | Пустой/битый `choices[0].message.content`.                                            |
| HTTP **200**, `status=error`, `error.code = text_too_long`          | Даже один чанк + SKILL не влезает.                                                    |
| HTTP **200**, `status=error`, `error.code = pipeline_error`         | Прочий сбой пайплайна.                                                                |
