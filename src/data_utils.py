"""Shared, immutable dataset contract and train/validation/test split."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / 'data/raw/Concrete_Data.xls'
SPLIT_FILE = ROOT / 'data/split.json'
SOURCE_URL = 'https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength'
MODEL_NAME = 'concrete-strength-regressor'
EXPERIMENT_NAME = 'lab4-concrete-strength'
FEATURES = [
    'cement',
    'blast_furnace_slag',
    'fly_ash',
    'water',
    'superplasticizer',
    'coarse_aggregate',
    'fine_aggregate',
    'age',
]
TARGET = 'compressive_strength'
UNITS = {**{feature: 'kg/m3' for feature in FEATURES[:-1]}, 'age': 'days'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset():
    """Return normalized data, preserving row order and raw file unchanged."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f'Missing {RAW_FILE}. Run: python src/prepare_data.py'
        )
    data = pd.read_excel(RAW_FILE, engine='xlrd')
    if data.shape[1] != 9:
        raise ValueError(f'Expected 9 columns (8 features + target), got {data.shape[1]}')
    data.columns = FEATURES + [TARGET]
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='raise')
    if data.isna().any().any():
        raise ValueError('Unexpected missing values in the UCI Concrete dataset')
    return data


def load_splits():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f'Missing {SPLIT_FILE}. Run: python src/prepare_data.py')
    info = json.loads(SPLIT_FILE.read_text(encoding='utf-8'))
    if info['raw_sha256'] != sha256(RAW_FILE):
        raise ValueError('Raw dataset hash changed: do not reuse the old split')
    df = load_dataset()
    if info['n_rows'] != len(df):
        raise ValueError('Number of rows changed')
    tr, va, te = [info[k] for k in ('train_indices', 'validation_indices', 'test_indices')]
    if len(set(tr + va + te)) != len(df) or set(tr + va + te) != set(range(len(df))):
        raise ValueError('Split is not a disjoint partition of the dataset')
    return df, info


def get_xy(df, indices):
    part = df.iloc[indices]
    return part[FEATURES], part[TARGET]
