FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# Results are written to /results; mount a volume to keep them.
RUN useradd --create-home runner && mkdir /results && chown runner /results
USER runner
VOLUME ["/results"]

ENTRYPOINT ["f1lap"]
CMD ["run", "--synthetic", "--out", "/results"]
