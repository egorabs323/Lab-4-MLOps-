"""Evaluate the selected frozen champion ONCE on untouched test data."""
import json
from pathlib import Path
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_utils import ROOT, get_xy, load_splits

df, split = load_splits()
meta = json.loads((ROOT / 'exported_model/metadata.json').read_text())
if meta['dataset_sha256'] != split['raw_sha256']:
    raise ValueError('Exported model references a different dataset')
model = mlflow.sklearn.load_model(str(ROOT / 'exported_model'))
X_test, y_test = get_xy(df, split['test_indices'])
y_pred = model.predict(X_test)
metrics = {
    'model_version': meta['version'], 'run_id': meta['run_id'],
    'n_test': len(X_test),
    'test_mae': float(mean_absolute_error(y_test, y_pred)),
    'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
}
(ROOT / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(json.dumps(metrics, indent=2))
