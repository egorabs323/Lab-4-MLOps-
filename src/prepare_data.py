"""Download original UCI ZIP and freeze one train/validation/test partition."""
import json
import urllib.request
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split
from data_utils import RAW_FILE, ROOT, SPLIT_FILE, SOURCE_URL, FEATURES, TARGET, load_dataset, sha256

ZIP_URL = 'https://archive.ics.uci.edu/static/public/165/concrete%2Bcompressive%2Bstrength.zip'
archive = ROOT / 'data/raw/concrete-compressive-strength-uci.zip'
RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

if not RAW_FILE.exists():
    if not archive.exists():
        print('Downloading original dataset from UCI...')
        try:
            urllib.request.urlretrieve(ZIP_URL, archive)
        except Exception as exc:
            raise SystemExit(
                'Could not download UCI ZIP automatically. Open the UCI source page, '
                'download the ZIP manually, extract Concrete_Data.xls into data/raw/ '
                'and rerun this script. Original error: ' + str(exc)
            )
    with zipfile.ZipFile(archive) as z:
        matches = [n for n in z.namelist() if Path(n).name.lower() == 'concrete_data.xls']
        if len(matches) != 1:
            raise RuntimeError(f'Concrete_Data.xls not uniquely found inside ZIP: {z.namelist()}')
        RAW_FILE.write_bytes(z.read(matches[0]))

frame = load_dataset()
print('Data shape:', frame.shape)
print('Missing values:', int(frame.isna().sum().sum()))
print('SHA-256:', sha256(RAW_FILE))

if SPLIT_FILE.exists():
    existing = json.loads(SPLIT_FILE.read_text(encoding='utf-8'))
    if existing['raw_sha256'] != sha256(RAW_FILE) or existing['n_rows'] != len(frame):
        raise SystemExit('data/split.json describes a different raw file; stop and inspect first.')
    print('Using existing split; train/val/test:', *(len(existing[k]) for k in
        ('train_indices', 'validation_indices', 'test_indices')))
else:
    idx = list(range(len(frame)))
    dev, test = train_test_split(idx, test_size=0.20, random_state=42)
    train, val = train_test_split(dev, test_size=0.25, random_state=42)
    split = {
        'source': SOURCE_URL, 'license': 'CC BY 4.0',
        'raw_file': 'data/raw/Concrete_Data.xls', 'raw_sha256': sha256(RAW_FILE),
        'random_state': 42, 'n_rows': len(frame), 'features': FEATURES, 'target': TARGET,
        'train_indices': train, 'validation_indices': val, 'test_indices': test,
    }
    SPLIT_FILE.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved split; train/val/test:', len(train), len(val), len(test))
