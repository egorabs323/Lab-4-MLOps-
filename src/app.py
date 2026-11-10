"""Local FastAPI regression service. Model is baked into Docker image."""
import json
import logging
import time
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title='Concrete strength regressor', version='1.0')
model = mlflow.sklearn.load_model('/app/exported_model')
meta = json.loads(Path('/app/exported_model/metadata.json').read_text())
log = logging.getLogger('uvicorn.error')


class PredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(min_length=1, max_length=100)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': meta['version']}


@app.get('/metadata')
def metadata():
    return meta


@app.post('/predict')
def predict(body: PredictRequest):
    start = time.perf_counter()
    expected = set(meta['features'])
    if any(set(row) != expected for row in body.instances):
        raise HTTPException(status_code=422, detail={
            'message': 'Each row must contain exactly these eight features',
            'features': meta['features'],
        })
    X = pd.DataFrame(body.instances)[meta['features']]
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise HTTPException(422, detail='All feature values must be finite')
    if (X.drop(columns=['age']) < 0).any().any():
        raise HTTPException(422, detail='Ingredient amounts must be non-negative')
    if ((X['age'] < 1) | (X['age'] > 365)).any():
        raise HTTPException(422, detail='Age must be within 1..365 days (dataset range)')
    pred = [float(v) for v in model.predict(X)]
    elapsed = round((time.perf_counter() - start) * 1000, 3)
    log.info('prediction model_version=%s rows=%s latency_ms=%.3f',
             meta['version'], len(X), elapsed)
    return {
        'predictions': pred, 'prediction_unit': 'MPa',
        'model_version': meta['version'], 'latency_ms': elapsed,
    }
