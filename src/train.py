"""Log one Ridge or RandomForestRegressor experiment to MLflow."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import (ROOT, EXPERIMENT_NAME, FEATURES, MODEL_NAME, SOURCE_URL,
                        SPLIT_FILE, get_xy, load_splits, sha256)

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', choices=['ridge', 'rf'], required=True)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--trees', type=int, default=100)
parser.add_argument('--depth', type=int, default=10)
parser.add_argument('--register', action='store_true')
a = parser.parse_args()
if a.alpha <= 0 or a.trees < 1 or a.depth < 1:
    parser.error('alpha, trees and depth must be positive')

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
mlflow.set_experiment(EXPERIMENT_NAME)
df, split = load_splits()
X_train, y_train = get_xy(df, split['train_indices'])
X_val, y_val = get_xy(df, split['validation_indices'])

if a.algorithm == 'ridge':
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          Ridge(alpha=a.alpha))
    params = {'alpha': a.alpha}
    run_name = f'ridge-alpha{a.alpha:g}'
else:
    model = make_pipeline(SimpleImputer(strategy='median'),
                          RandomForestRegressor(n_estimators=a.trees,
                                                max_depth=a.depth,
                                                random_state=42, n_jobs=-1))
    params = {'trees': a.trees, 'depth': a.depth}
    run_name = f'rf-trees{a.trees}-depth{a.depth}'

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params({
        'algorithm': a.algorithm, 'seed': 42, 'dataset': 'UCI Concrete 165',
        'n_train': len(X_train), 'n_validation': len(X_val),
        'n_test': len(split['test_indices']), **params,
    })
    mlflow.set_tags({
        'dataset_sha256': split['raw_sha256'],
        'split_sha256': sha256(SPLIT_FILE),
        'purpose': 'lab4-variant14', 'selection_metric': 'val_mae',
        'source': SOURCE_URL,
    })
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)
    metrics = {
        'val_mae': float(mean_absolute_error(y_val, val_pred)),
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
        'train_mae': float(mean_absolute_error(y_train, train_pred)),
        'fit_seconds': fit_seconds,
    }
    mlflow.log_metrics(metrics)
    mlflow.log_input(
        mlflow.data.from_pandas(df, source=SOURCE_URL, targets='compressive_strength'),
        context='full_dataset_before_split',
    )
    mlflow.log_artifact(str(SPLIT_FILE), artifact_path='reproducibility')
    mlflow.log_artifact(__file__, artifact_path='source')
    mlflow.log_artifact(str(ROOT / 'src/data_utils.py'), artifact_path='source')
    mlflow.log_artifact(str(ROOT / 'requirements.txt'), artifact_path='source')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_val, val_pred, s=18, alpha=0.7)
    minimum = min(float(y_val.min()), float(np.min(val_pred)))
    maximum = max(float(y_val.max()), float(np.max(val_pred)))
    ax.plot([minimum, maximum], [minimum, maximum], '--', linewidth=1)
    ax.set_xlabel('Observed strength, MPa')
    ax.set_ylabel('Predicted strength, MPa')
    ax.set_title('Validation: measured vs predicted')
    fig.tight_layout()
    mlflow.log_figure(fig, 'validation_actual_vs_predicted.png')
    plt.close(fig)
    info = mlflow.sklearn.log_model(
        model, name='model', input_example=X_train.head(2),
        signature=infer_signature(X_train, model.predict(X_train)),
        pip_requirements=[x for x in (ROOT / 'requirements.txt').read_text().splitlines() if x],
    )
    print(json.dumps({'run_id': run.info.run_id, 'model_uri': info.model_uri,
                      **metrics}, indent=2))
    if a.register:
        version = mlflow.register_model(info.model_uri, MODEL_NAME)
        print('REGISTERED_VERSION=', version.version)
