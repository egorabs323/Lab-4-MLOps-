"""Smoke test: correct request, model agreement and invalid payloads."""
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8000'


def post(payload):
    data = json.dumps(payload, allow_nan=False).encode('utf-8')
    req = urllib.request.Request(BASE + '/predict', data=data,
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


sample = json.loads((ROOT / 'sample_request.json').read_text())
expected = json.loads((ROOT / 'expected_predictions.json').read_text())
meta = json.loads((ROOT / 'exported_model/metadata.json').read_text())
status, result = post(sample)
assert status == 200, (status, result)
assert len(result['predictions']) == len(expected)
assert result['model_version'] == str(meta['version'])
assert all(math.isclose(float(a), float(b), abs_tol=1e-6, rel_tol=1e-9)
           for a, b in zip(result['predictions'], expected)), (result, expected)
print('PASS: container predictions match exported model:', result['predictions'])

wrong_cases = {
    'missing features': {'instances': [{'cement': 540.0}]},
    'empty batch': {'instances': []},
    'extra field': {'instances': [{**sample['instances'][0], 'unknown': 123}]},
    'wrong type': {'instances': [{**sample['instances'][0], 'cement': 'not_a_number'}]},
}
for title, payload in wrong_cases.items():
    code, _ = post(payload)
    assert code == 422, (title, code)
    print(f'PASS: {title} -> HTTP 422')

status, _ = post({'instances': [sample['instances'][0]]})
assert status == 200
print('PASS: single valid instance -> HTTP 200')
