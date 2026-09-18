FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/data \
    SQLITE_PATH=/data/tasks.db \
    LOG_DIR=/data/logs \
    PERFORMANCE_LOG=/data/logs/performance_log.csv \
    WORKERS=1

RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir -U pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY version.txt ./
COPY app ./app

RUN set -eux; \
    if ! getent group 1001 >/dev/null; then groupadd --gid 1001 app; fi; \
    if ! getent passwd 1001 >/dev/null; then \
        useradd --uid 1001 --gid 1001 --create-home --home-dir /home/app --shell /usr/sbin/nologin app; \
    fi; \
    mkdir -p /data; \
    chown -R 1001:1001 /app /opt/venv /data

USER 1001:1001

EXPOSE 8000
VOLUME ["/data"]

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers 1"]
