FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

WORKDIR /app

RUN addgroup --gid 1000 pulse \
    && adduser --uid 1000 --gid 1000 --disabled-password --gecos "" pulse

COPY deploy/pulse/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY src/sphinx_pulse /app/sphinx_pulse
COPY deploy/pulse/pulse.yaml /app/pulse.yaml

USER pulse

CMD ["python", "-m", "sphinx_pulse.collector", "--config", "/app/pulse.yaml"]
