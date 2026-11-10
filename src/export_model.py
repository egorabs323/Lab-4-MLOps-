"""Export champion as a snapshot and create request/expected response fixtures."""
import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from data_utils import (ROOT, MODEL_NAME, SOURCE_URL, FEATURES, TARGET, UNITS,
                        get_xy, load_splits)

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
version = MlflowClient().get_model_version_by_alias(MODEL_NAME, 'champion')
model = mlflow.sklearn.load_model(f'models:/{MODEL_NAME}/{version.version}')
out = ROOT / 'exported_model'
if out.exists():
    shutil.rmtree(out)
mlflow.sklearn.save_model(
    model, str(out),
    pip_requirements=(ROOT / 'requirements.txt').read_text().splitlines(),
)
df, split = load_splits()
X_test, _ = get_xy(df, split['test_indices'])
metadata = {
    'name': MODEL_NAME, 'version': str(version.version), 'run_id': version.run_id,
    'dataset_source': SOURCE_URL, 'dataset_sha256': split['raw_sha256'],
    'features': FEATURES, 'feature_units': UNITS,
    'target': TARGET, 'target_unit': 'MPa',
    'feature_description': 'Ingredient amounts per 1 m3 of mixture; age in days',
    'age_range_in_dataset': '1..365 days',
}
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
sample = X_test.iloc[:3]
(ROOT / 'sample_request.json').write_text(
    json.dumps({'instances': sample.to_dict(orient='records')}, indent=2),
    encoding='utf-8',
)
(ROOT / 'expected_predictions.json').write_text(
    json.dumps([float(x) for x in model.predict(sample)], indent=2), encoding='utf-8',
)
print(json.dumps(metadata, indent=2))
