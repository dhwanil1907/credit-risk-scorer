FROM python:3.13-slim AS build

WORKDIR /wheels
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home --shell /usr/sbin/nologin scorer

WORKDIR /app
COPY --from=build /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=scorer:scorer api.py app.py ./
COPY --chown=scorer:scorer src ./src
COPY --chown=scorer:scorer models/xgboost.pkl ./models/xgboost.pkl
COPY --chown=scorer:scorer outputs/feature_cols.json outputs/metrics.csv outputs/shap_summary.png ./outputs/

USER scorer
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
