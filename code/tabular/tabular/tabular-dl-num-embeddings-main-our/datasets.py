# %%
import argparse
import enum
import json
import math
import random
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, cast
from urllib.request import urlretrieve
import os
import catboost.datasets
import geopy.distance
import numpy as np
import pandas as pd
import pyarrow.csv
import sklearn.datasets
import sklearn.utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

ArrayDict = dict[str, np.ndarray]
Info = dict[str, Any]

print("hoem", Path.home())
SEED = 0
CAT_MISSING_VALUE = '__nan__'

EXPECTED_FILES = {
    'eye': [],
    'gas': [],
    'gesture': [],
    'house': [],
    'higgs-small': [],
    # Run `kaggle competitions download -c santander-customer-transaction-prediction`
    'santander': ['santander-customer-transaction-prediction.zip'],
    # Run `kaggle competitions download -c otto-group-product-classification-challenge`
    'otto': ['otto-group-product-classification-challenge.zip'],
    # Run `kaggle competitions download -c rossmann-store-sales`
    'rossmann': ['rossmann-store-sales.zip'],
    # Source: https://www.kaggle.com/shrutimechlearn/churn-modelling
    'churn': ['Churn_Modelling.csv'],
    # Source: https://www.kaggle.com/neomatrix369/nyc-taxi-trip-duration-extended
    'taxi': ['train_extended.csv.zip'],
    # Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip
    'fb-comments': ['Dataset.zip'],
    'california': [],
    'covtype': [],
    'adult': [],
    # Source: https://www.dropbox.com/s/572rj8m5f9l2nz5/MSLR-WEB10K.zip?dl=1
    # This is literally the official data, but reuploded to Dropbox.
    'microsoft': ['MSLR-WEB10K.zip'],
}
EXPECTED_FILES['wd-taxi'] = EXPECTED_FILES['taxi']
EXPECTED_FILES['fb-c'] = EXPECTED_FILES['wd-fb-comments'] = EXPECTED_FILES[
    'fb-comments'
]


class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'


# %%
def _set_random_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


def _download_file(url: str, path: Path):
    assert not path.exists()
    try:
        print(f'Downloading {url} ...', end='', flush=True)
        urlretrieve(url, path)
    except Exception:
        if path.exists():
            path.unlink()
        raise
    finally:
        print()


def _unzip(path: Path, members: Optional[list[str]] = None) -> None:
    with zipfile.ZipFile(path) as f:
        f.extractall(path.parent, members)


# def _start(dirname: str) -> tuple[Path, list[Path]]:
#     print(f'>>> {dirname}')
#     _set_random_seeds()
#     dataset_dir = DATA_DIR / dirname
#     expected_files = EXPECTED_FILES[dirname]
#     if expected_files:
#         assert dataset_dir.exists()
#         assert set(expected_files) == set(x.name for x in dataset_dir.iterdir())
#     else:
#         assert not dataset_dir.exists()
#         dataset_dir.mkdir()
#     return dataset_dir, [dataset_dir / x for x in expected_files]


def _fetch_openml(data_id: int) -> sklearn.utils.Bunch:
    bunch = cast(
        sklearn.utils.Bunch,
        sklearn.datasets.fetch_openml(data_id=data_id, as_frame=True),
    )
    assert not bunch['categories']
    return bunch


def _get_sklearn_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    get_data = getattr(sklearn.datasets, f'load_{name}', None)
    if get_data is None:
        get_data = getattr(sklearn.datasets, f'fetch_{name}', None)
    assert get_data is not None, f'No such dataset in scikit-learn: {name}'
    return get_data(return_X_y=True)


def _encode_classification_target(y: np.ndarray) -> np.ndarray:
    assert not str(y.dtype).startswith('float')
    if str(y.dtype) not in ['int32', 'int64', 'uint32', 'uint64']:
        y = LabelEncoder().fit_transform(y)
    else:
        labels = set(map(int, y))
        if sorted(labels) != list(range(len(labels))):
            y = LabelEncoder().fit_transform(y)
    return y.astype(np.int64)


# def _make_split(size: int, stratify: Optional[np.ndarray], n_parts: int) -> ArrayDict:
#     # n_parts == 3:      all -> train & val & test
#     # n_parts == 2: trainval -> train & val
#     assert n_parts in (2, 3)
#     all_idx = np.arange(size, dtype=np.int64)
#     a_idx, b_idx = train_test_split(
#         all_idx,
#         test_size=0.2,
#         stratify=stratify,
#         random_state=SEED + (1 if n_parts == 2 else 0),
#     )
#     if n_parts == 2:
#         return cast(ArrayDict, {'train': a_idx, 'val': b_idx})
#     a_stratify = None if stratify is None else stratify[a_idx]
#     a1_idx, a2_idx = train_test_split(
#         a_idx, test_size=0.2, stratify=a_stratify, random_state=SEED + 1
#     )
#     X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
#     train_index = X[X.Set != "test"].index
#     test_index = X[X.Set == "test"].index
#     return cast(ArrayDict, {'train': a1_idx, 'val': a2_idx, 'test': b_idx})


def _apply_split(data: ArrayDict, split: ArrayDict) -> dict[str, ArrayDict]:
    return {k: {part: v[idx] for part, idx in split.items()} for k, v in data.items()}


def _save(
        dataset_dir: Path,
        name: str,
        task_type: TaskType,
        *,
        X_num: Optional[ArrayDict] = None,
        X_cat: Optional[ArrayDict] = None,
        y: ArrayDict,
        idx: Optional[ArrayDict],
        id_: Optional[str] = None,
        id_suffix: str = '--default',
) -> None:
    if id_ is not None:
        assert id_suffix == '--default'
    assert (
            X_num is not None or X_cat is not None
    ), 'At least one type of features must be presented.'
    if X_num is not None:
        X_num = {k: v.astype(np.float32) for k, v in X_num.items()}
    if X_cat is not None:
        X_cat = {k: v.astype(str) for k, v in X_cat.items()}
    if idx is not None:
        idx = {k: v.astype(np.int64) for k, v in idx.items()}
    y = {
        k: v.astype(np.float32 if task_type == TaskType.REGRESSION else np.int64)
        for k, v in y.items()
    }
    if task_type != TaskType.REGRESSION:
        y_unique = {k: set(v.tolist()) for k, v in y.items()}
        assert y_unique['train'] == set(range(max(y_unique['train']) + 1))
        for x in ['val', 'test']:
            assert y_unique[x] <= y_unique['train']
        del x

    info = {
               'name': name,
               'id': (dataset_dir.name + id_suffix) if id_ is None else id_,
               'task_type': task_type.value,
               'n_num_features': (0 if X_num is None else next(iter(X_num.values())).shape[1]),
               'n_cat_features': (0 if X_cat is None else next(iter(X_cat.values())).shape[1]),
           } | {f'{k}_size': len(v) for k, v in y.items()}
    if task_type == TaskType.MULTICLASS:
        info['n_classes'] = len(set(y['train']))
    (dataset_dir / 'info.json').write_text(json.dumps(info, indent=4))

    for data_name in ['X_num', 'X_cat', 'y', 'idx']:
        data = locals()[data_name]
        if data is not None:
            for k, v in data.items():
                np.save(dataset_dir / f'{data_name}_{k}.npy', v)
    (dataset_dir / 'READY').touch()
    print('Done\n')

def _save_label(
        dataset_dir: Path,
        name: str,
        task_type: TaskType,
        *,
        X_num: Optional[ArrayDict] = None,
        X_cat: Optional[ArrayDict] = None,
        y: ArrayDict,
        id_: Optional[str] = None,
        id_suffix: str = '--default',
) -> None:
    if id_ is not None:
        assert id_suffix == '--default'
    assert (
            X_num is not None or X_cat is not None
    ), 'At least one type of features must be presented.'
    if X_num is not None:
        X_num = {k: v.astype(np.float32) for k, v in X_num.items()}
    if X_cat is not None:
        X_cat = {k: v.astype(str) for k, v in X_cat.items()}

    y = {
        k: v.astype(np.float32 if task_type == TaskType.REGRESSION else np.int64)
        for k, v in y.items()
    }
    if task_type != TaskType.REGRESSION:
        y_unique = {k: set(v.tolist()) for k, v in y.items()}
        assert y_unique['train'] == set(range(max(y_unique['train']) + 1))
        for x in ['val', 'test']:
            assert y_unique[x] <= y_unique['train']
        del x

    info = {
               'name': name,
               'id': (dataset_dir.name + id_suffix) if id_ is None else id_,
               'task_type': task_type.value,
               'n_num_features': (0 if X_num is None else next(iter(X_num.values())).shape[1]),
               'n_cat_features': (0 if X_cat is None else next(iter(X_cat.values())).shape[1]),
           } | {f'{k}_size': len(v) for k, v in y.items()}
    if task_type == TaskType.MULTICLASS:
        info['n_classes'] = len(set(y['train']))
    (dataset_dir / 'info.json').write_text(json.dumps(info, indent=4))

    for data_name in ['X_num', 'X_cat', 'y']:
        data = locals()[data_name]
        if data is not None:
            for k, v in data.items():
                np.save(dataset_dir / f'{data_name}_{k}.npy', v)
    (dataset_dir / 'READY').touch()
    print('Done\n')
# %%
# def eye_movements():
#     dataset_dir, _ = _start('eye')
#     bunch = _fetch_openml(1044)
#
#     X_num_all = bunch['data'].drop(columns=['lineNo']).values.astype(np.float32)
#     y_all = _encode_classification_target(bunch['target'].cat.codes.values)
#     idx = _make_split(len(X_num_all), y_all, 3)
#
#     _save(
#         dataset_dir,
#         'Eye Movements',
#         TaskType.MULTICLASS,
#         **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
#         X_cat=None,
#         idx=idx,
#     )
#
#
# def gesture_phase():
#     dataset_dir, _ = _start('gesture')
#     bunch = _fetch_openml(4538)
#
#     X_num_all = bunch['data'].values.astype(np.float32)
#     y_all = _encode_classification_target(bunch['target'].cat.codes.values)
#     idx = _make_split(len(X_num_all), y_all, 3)
#
#     _save(
#         dataset_dir,
#         'Gesture Phase',
#         TaskType.MULTICLASS,
#         **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
#         X_cat=None,
#         idx=idx,
#     )
#
#
# def house_16h():
#     dataset_dir, _ = _start('house')
#     bunch = _fetch_openml(574)
#
#     X_num_all = bunch['data'].values.astype(np.float32)
#     y_all = bunch['target'].values.astype(np.float32)
#     idx = _make_split(len(X_num_all), None, 3)
#
#     _save(
#         dataset_dir,
#         'House 16H',
#         TaskType.REGRESSION,
#         **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
#         X_cat=None,
#         idx=idx,
#     )
#
#
# def higgs_small():
#     dataset_dir, _ = _start('higgs-small')
#     bunch = _fetch_openml(23512)
#
#     X_num_all = bunch['data'].values.astype(np.float32)
#     y_all = _encode_classification_target(bunch['target'].cat.codes.values)
#     nan_mask = np.isnan(X_num_all)
#     valid_objects_mask = ~(nan_mask.any(1))
#     # There is just one object with nine(!) missing values; let's drop it
#     assert valid_objects_mask.sum() + 1 == len(X_num_all) and nan_mask.sum() == 9
#     X_num_all = X_num_all[valid_objects_mask]
#     y_all = y_all[valid_objects_mask]
#     idx = _make_split(len(X_num_all), y_all, 3)
#
#     _save(
#         dataset_dir,
#         'Higgs Small',
#         TaskType.BINCLASS,
#         **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
#         X_cat=None,
#         idx=idx,
#     )
#
#
# def santander_customer_transactions():
#     dataset_dir, files = _start('santander')
#     _unzip(files[0])
#
#     df = pd.read_csv(dataset_dir / 'train.csv')
#     df.drop(columns=['ID_code'], inplace=True)
#     y_all = _encode_classification_target(df.pop('target').values)  # type: ignore[code]
#     X_num_all = df.values.astype(np.float32)
#     idx = _make_split(len(X_num_all), y_all, 3)
#
#     _save(
#         dataset_dir,
#         'Santander Customer Transactions',
#         TaskType.BINCLASS,
#         **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
#         X_cat=None,
#         idx=idx,
#     )
#
#
# def otto_group_products():
#     dataset_dir, files = _start('otto')
#     _unzip(files[0])
#
#     df = pd.read_csv(dataset_dir / 'train.csv')
#     df.drop(columns=['id'], inplace=True)
#     y_all = _encode_classification_target(
#         df.pop('target').map(lambda x: int(x.split('_')[-1]) - 1).values  # type: ignore[code]
#     )
#     X_num_all = df.values.astype(np.float32)
#     idx = _make_split(len(X_num_all), y_all, 3)
#
#     _save(
#         dataset_dir,
#         'Otto Group Products',
#         TaskType.MULTICLASS,
#         **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
#         X_cat=None,
#         idx=idx,
#     )
#
#
# def churn_modelling():
#     # Get the file here: https://www.kaggle.com/shrutimechlearn/churn-modelling
#     dataset_dir, files = _start('churn')
#     df = pd.read_csv(files[0])
#
#     df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
#     df['Gender'] = df['Gender'].astype('category').cat.codes.values.astype(np.int64)
#     y_all = df.pop('Exited').values.astype(np.int64)
#     num_columns = [
#         'CreditScore',
#         'Gender',
#         'Age',
#         'Tenure',
#         'Balance',
#         'NumOfProducts',
#         'EstimatedSalary',
#         'HasCrCard',
#         'IsActiveMember',
#         'EstimatedSalary',
#     ]
#     cat_columns = ['Geography']
#     assert set(num_columns) | set(cat_columns) == set(df.columns.tolist())
#     X_num_all = df[num_columns].astype(np.float32).values
#     X_cat_all = df[cat_columns].astype(str).values
#     idx = _make_split(len(df), y_all, 3)
#
#     _save(
#         dataset_dir,
#         'Churn Modelling',
#         TaskType.BINCLASS,
#         **_apply_split(
#             {'X_num': X_num_all, 'X_cat': X_cat_all, 'y': y_all},
#             idx,
#         ),
#         idx=idx,
#     )
#

# def News():
#     datafile = "/data/lsw/missingdata/data/News"
#     _set_random_seeds()
#     # dataset_dir = "data/HI"
#     missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     for rate in missing_rates:
#         dataset_dir = "data/News_" + str(rate)
#         df = pd.read_csv(datafile + "/mcar_News_" + str(rate) + ".csv")
#         y_all = _encode_classification_target(df.pop('shares').values)  # type: ignore[code]
#         X_num_all = df.values.astype(np.float32)
#         idx = _make_split(len(X_num_all), y_all, 3)
#     # X_num_all, y_all = _get_sklearn_dataset('california_housing')
#     # idx = _make_split(len(X_num_all), None, 3)
#
#     _save(
#         dataset_dir,
#         'News',
#         TaskType.BINCLASS,
#         **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
#         X_cat=None,
#         idx=idx,
#     )
def HI():
    # dataset_dir, _ = _start('HI')

    # dataset_dir = "data/HI"
    miss_types = ["mnar_p_", "mar_p_"]
    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    imputations = ["mean"]
    train_index = []
    valid_index = []
    test_index = []
    for rate in missing_rates:
        for miss_type in miss_types:
            for imp in imputations:
                _set_random_seeds()


                if imp is not "mean":
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/HI/" + miss_type + imp + str(
                            rate))
                    out = Path("/data/lsw/data/data/HI/" + miss_type + "HI_" + str(
                    rate) + imp + ".csv")
                else:
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/HI/" + miss_type + str(
                            rate))
                    out = Path("/data/lsw/data/data/HI/" + miss_type + "HI_" + str(
                        rate) + ".csv")
                if not os.path.exists(dataset_dir):
                    os.mkdir(dataset_dir)
                df = pd.read_csv(out)
                y_all = _encode_classification_target(df.pop('Labels').values)  # type: ignore[code]
                X = df
                nunique = X.nunique()
                types = X.dtypes
                categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
                for col in X.columns:
                    if types[col] == 'object' or nunique[col] < 100:
                        categorical_indicator[X.columns.get_loc(col)] = True

                categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

                cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                X_num_all = df.iloc[:, con_idxs].astype(np.float32).values
                X_cat_all = df.iloc[:, cat_idxs].astype(str).values
                if (len(train_index) == 0):
                    X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
                    train_index = X[X.Set == "train"].index
                    valid_index = X[X.Set == "valid"].index
                    test_index = X[X.Set == "test"].index
                    X = X.drop(columns=['Set']).to_numpy()
                idx = cast(ArrayDict, {'train': train_index, 'val': valid_index, 'test': test_index})
                # X_num_all, y_all = _get_sklearn_dataset('california_housing')
                # idx = _make_split(len(X_num_all), None, 3)
                try:
                    _save(
                        dataset_dir,
                        'HI',
                        TaskType.BINCLASS,
                        **_apply_split({'X_num': X_num_all, 'X_cat': X_cat_all, 'y': y_all}, idx),
                        idx=idx,
                    )
                except:
                    _save(
                        dataset_dir,
                        'HI',
                        TaskType.BINCLASS,
                        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
                        idx=idx,
                    )


def Letter():
    # dataset_dir, _ = _start('HI')

    # dataset_dir = "data/HI"
    miss_types = ["mnar_", "mcar_", "mar_"]
    missing_rates = [0.5]
    imputations = ["_gain"]
    train_index = []
    valid_index = []
    test_index = []
    for rate in missing_rates:
        for miss_type in miss_types:
            for imp in imputations:
                _set_random_seeds()
                dataset_dir = Path(
                    "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/Letter/" + miss_type + imp + str(
                        rate))
                if not os.path.exists(dataset_dir):
                    os.mkdir(dataset_dir)
                out = Path("/data/lsw/data/data/Letter/" + miss_type + "Letter_" + str(
                    rate) + imp + ".csv")
                df = pd.read_csv(out)
                y_all = _encode_classification_target(df.pop('Labels').values)  # type: ignore[code]
                X = df
                nunique = X.nunique()
                types = X.dtypes
                categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
                for col in X.columns:
                    if types[col] == 'object' or nunique[col] < 100:
                        categorical_indicator[X.columns.get_loc(col)] = True

                categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

                cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                X_num_all = df.iloc[:, con_idxs].astype(np.float32).values
                X_cat_all = df.iloc[:, cat_idxs].astype(str).values
                if (len(train_index) == 0):
                    X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
                    train_index = X[X.Set == "train"].index
                    valid_index = X[X.Set == "valid"].index
                    test_index = X[X.Set == "test"].index
                    X = X.drop(columns=['Set']).to_numpy()
                idx = cast(ArrayDict, {'train': train_index, 'val': valid_index, 'test': test_index})
                # X_num_all, y_all = _get_sklearn_dataset('california_housing')
                # idx = _make_split(len(X_num_all), None, 3)

                _save(
                    dataset_dir,
                    'Letter',
                    TaskType.MULTICLASS,
                    **_apply_split({'X_cat': X_cat_all, 'y': y_all}, idx),
                    idx=idx,
                )


def temperature():
    # dataset_dir, _ = _start('HI')

    # dataset_dir = "data/HI"
    miss_types = ["mnar_p_", "mar_p_"]
    missing_rates = [0.5]
    imputations = ["mean"]
    train_index = []
    valid_index = []
    test_index = []
    for rate in missing_rates:
        for miss_type in miss_types:
            for imp in imputations:
                _set_random_seeds()

                if imp is not "mean":
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/temperature/" + miss_type + imp + str(
                            rate))
                    out = Path("/data/lsw/data/data/temperature/" + miss_type + "temperature_" + str(
                        rate) + imp + ".csv")
                else:
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/temperature/" + miss_type + str(
                            rate))
                    out = Path("/data/lsw/data/data/temperature/" + miss_type + "temperature_" + str(
                        rate) + ".csv")
                if not os.path.exists(dataset_dir):
                    os.mkdir(dataset_dir)
                df = pd.read_csv(out)
                y_all = df['target'].values.astype(np.float32)  # type: ignore[code]
                df.pop('target').values
                X = df
                nunique = X.nunique()
                types = X.dtypes
                categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
                for col in X.columns:
                    if types[col] == 'object' or nunique[col] < 100:
                        categorical_indicator[X.columns.get_loc(col)] = True

                categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

                cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                X_num_all = df.iloc[:, con_idxs].astype(np.float32).values
                X_cat_all = df.iloc[:, cat_idxs].astype(str).values
                if (len(train_index) == 0):
                    X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
                    train_index = X[X.Set == "train"].index
                    valid_index = X[X.Set == "valid"].index
                    test_index = X[X.Set == "test"].index
                    X = X.drop(columns=['Set']).to_numpy()
                idx = cast(ArrayDict, {'train': train_index, 'val': valid_index, 'test': test_index})
                # X_num_all, y_all = _get_sklearn_dataset('california_housing')
                # idx = _make_split(len(X_num_all), None, 3)

                _save(
                    dataset_dir,
                    'temperature',
                    TaskType.REGRESSION,
                    **_apply_split({'X_num': X_num_all, 'X_cat': X_cat_all,  'y': y_all}, idx),
                    idx=idx,
                )


def gas():
    # dataset_dir, _ = _start('HI')

    # dataset_dir = "data/HI"
    miss_types = ["mcar_", "mnar_p_", "mar_p_"]
    missing_rates = [0.1, 0.3, 0.7, 0.9]
    imputations = [ "mean"]
    train_index = []
    valid_index = []
    test_index = []
    for rate in missing_rates:
        for miss_type in miss_types:
            for imp in imputations:
                _set_random_seeds()
                if imp == "mean":
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/gas/" + miss_type + str(
                            rate))
                    out = Path("/data/lsw/data/data/gas/" + miss_type + "gas_" + str(
                        rate) + ".csv")
                else:
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/gas/" + miss_type + imp + str(
                            rate))
                    out = Path("/data/lsw/data/data/gas/" + miss_type + "gas_" + str(
                        rate) + imp + ".csv")
                if not os.path.exists(dataset_dir):
                    os.mkdir(dataset_dir)
                df = pd.read_csv(out)
                y_all = df['target'].values.astype(np.float32)  # type: ignore[code]
                df.pop('target').values
                X = df
                nunique = X.nunique()
                types = X.dtypes
                categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
                for col in X.columns:
                    if types[col] == 'object' or nunique[col] < 100:
                        categorical_indicator[X.columns.get_loc(col)] = True

                categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

                cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                X_num_all = df.iloc[:, con_idxs].astype(np.float32).values
                X_cat_all = df.iloc[:, cat_idxs].astype(str).values
                if (len(train_index) == 0):
                    X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
                    train_index = X[X.Set == "train"].index
                    valid_index = X[X.Set == "valid"].index
                    test_index = X[X.Set == "test"].index
                    X = X.drop(columns=['Set']).to_numpy()
                idx = cast(ArrayDict, {'train': train_index, 'val': valid_index, 'test': test_index})
                # X_num_all, y_all = _get_sklearn_dataset('california_housing')
                # idx = _make_split(len(X_num_all), None, 3)

                _save(
                    dataset_dir,
                    'gas',
                    TaskType.REGRESSION,
                    **_apply_split({'X_num': X_num_all,'X_cat': X_cat_all, 'y': y_all}, idx),
                    idx=idx,
                )


def higgs():
    # dataset_dir = "data/HI"
    miss_types = ["mnar_", "mcar_", "mar_"]
    missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_index = []
    valid_index = []
    test_index = []
    for rate in missing_rates:
        for miss_type in miss_types:
            _set_random_seeds()
            dataset_dir = Path(
                "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/HIGGS/" + miss_type + str(rate))
            if not os.path.exists(dataset_dir):
                os.mkdir(dataset_dir)
            out = Path("/data/lsw/representations/data/HIGGS/" + miss_type + "HIGGS_" + str(
                rate) + ".csv")
            df = pd.read_csv(out)
            asd = df.nunique()
            # print(asd)
            y_all = _encode_classification_target(df.pop('Labels').values.astype(int))  # type: ignore[code]
            X = df
            nunique = X.nunique()
            types = X.dtypes
            categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
            for col in X.columns:
                if types[col] == 'object' or nunique[col] < 100:
                    categorical_indicator[X.columns.get_loc(col)] = True

            categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
            cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

            cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
            con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
            X_num_all = df.iloc[:, con_idxs].astype(np.float32).values
            X_cat_all = df.iloc[:, cat_idxs].astype(str).values
            if (len(train_index) == 0):
                X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
                train_index = X[X.Set == "train"].index
                valid_index = X[X.Set == "valid"].index
                test_index = X[X.Set == "test"].index
                X = X.drop(columns=['Set']).to_numpy()
            idx = cast(ArrayDict, {'train': train_index, 'val': valid_index, 'test': test_index})
            # X_num_all, y_all = _get_sklearn_dataset('california_housing')
            # idx = _make_split(len(X_num_all), None, 3)

            _save(
                dataset_dir,
                'HIGGS',
                TaskType.MULTICLASS,
                **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
                idx=idx,
            )


def Gesture():
    # dataset_dir, _ = _start('HI')

    # dataset_dir = "data/HI"
    miss_types = ["mnar_p_", "mcar_", "mar_p_"]
    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    imputations = ["mean", "_notmiwae", "_miwae", "_gain", "_missforest"]
    train_index = []
    valid_index = []
    test_index = []
    for rate in missing_rates:
        for miss_type in miss_types:
            for imp in imputations:
                _set_random_seeds()
                if imp is not "mean":
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/Gesture/" + miss_type + imp + str(
                            rate))
                    out = Path("/data/lsw/data/data/Gesture/" + miss_type + "Gesture_" + str(
                        rate) + imp + ".csv")
                else:
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/Gesture/" + miss_type + str(
                            rate))
                    out = Path("/data/lsw/data/data/Gesture/" + miss_type + "Gesture_" + str(
                        rate) + ".csv")
                if not os.path.exists(dataset_dir):
                    os.mkdir(dataset_dir)
                df = pd.read_csv(out)
                y_all = _encode_classification_target(df.pop('Labels').values)  # type: ignore[code]
                X = df
                nunique = X.nunique()
                types = X.dtypes
                categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
                for col in X.columns:
                    if types[col] == 'object' or nunique[col] < 100:
                        categorical_indicator[X.columns.get_loc(col)] = True

                categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

                cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                X_num_all = df.iloc[:, con_idxs].astype(np.float32).values
                X_cat_all = df.iloc[:, cat_idxs].astype(str).values
                if (len(train_index) == 0):
                    X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
                    train_index = X[X.Set == "train"].index
                    valid_index = X[X.Set == "valid"].index
                    test_index = X[X.Set == "test"].index
                    X = X.drop(columns=['Set']).to_numpy()
                idx = cast(ArrayDict, {'train': train_index, 'val': valid_index, 'test': test_index})
                # X_num_all, y_all = _get_sklearn_dataset('california_housing')
                # idx = _make_split(len(X_num_all), None, 3)

                _save(
                    dataset_dir,
                    'Gesture',
                    TaskType.MULTICLASS,
                    **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
                    idx=idx,
                )


def News():
    # dataset_dir, _ = _start('HI')

    # dataset_dir = "data/HI"
    miss_types = ["mcar_label_"]
    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    imputations = ["_missforest"]
    train_index = []
    valid_index = []
    test_index = []
    for rate in missing_rates:
        for miss_type in miss_types:
            for imp in imputations:
                _set_random_seeds()
                if imp is not "mean":
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/News/" + miss_type + imp + str(
                            rate))
                    out = Path("/data/lsw/data/data/News/" + miss_type + "News_" + str(
                        rate) + imp + ".csv")
                else:
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/News/" + miss_type + str(
                            rate))
                    out = Path("/data/lsw/data/data/News/" + miss_type + "News_" + str(
                        rate) + ".csv")
                if not os.path.exists(dataset_dir):
                    os.mkdir(dataset_dir)
                df = pd.read_csv(out)
                y_all = df.pop('shares').values  # type: ignore[code]
                X = df
                nunique = X.nunique()
                types = X.dtypes
                categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
                for col in X.columns:
                    if types[col] == 'object' or nunique[col] < 100:
                        categorical_indicator[X.columns.get_loc(col)] = True

                categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

                cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                X_num_all = df.iloc[:, con_idxs].astype(np.float32).values
                X_cat_all = df.iloc[:, cat_idxs].astype(str).values
                if (len(train_index) == 0):
                    X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
                    train_index = X[X.Set == "train"].index
                    valid_index = X[X.Set == "valid"].index
                    test_index = X[X.Set == "test"].index
                    X = X.drop(columns=['Set']).to_numpy()
                combined_array = np.hstack((X[train_index], y_all[train_index].reshape(-1, 1)))
                # 根据最后一列是否为空来过滤样本
                non_empty_rows = combined_array[~np.isnan(combined_array[:, -1])]
                idx = cast(ArrayDict, {'train': train_index, 'val': valid_index, 'test': test_index})
                # X_num_all, y_all = _get_sklearn_dataset('california_housing')
                # idx = _make_split(len(X_num_all), None, 3)

                _save(
                    dataset_dir,
                    'News',
                    TaskType.BINCLASS,
                    **_apply_split({'X_num': X_num_all, 'X_cat': X_cat_all,  'y': y_all}, idx),
                    idx=idx,
                )

def News_label():
    # dataset_dir, _ = _start('HI')

    # dataset_dir = "data/HI"
    miss_types = ["mcar_label_"]
    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    imputations = ["_missforest"]
    train_index = []
    valid_index = []
    test_index = []
    for rate in missing_rates:
        for miss_type in miss_types:
            for imp in imputations:
                _set_random_seeds()
                if imp != "mean":
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/News/" + miss_type + imp + str(
                            rate))
                    out = Path("/data/lsw/data/data/News/" + miss_type + "News_" + str(
                        rate) + imp + ".csv")
                else:
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/News/" + miss_type + str(
                            rate))
                    out = Path("/data/lsw/data/data/News/" + miss_type + "News_" + str(
                        rate) + ".csv")
                if not os.path.exists(dataset_dir):
                    os.mkdir(dataset_dir)
                df = pd.read_csv(out)
                y_all = df.pop('shares').values  # type: ignore[code]
                X = df
                nunique = X.nunique()
                types = X.dtypes
                categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
                for col in X.columns:
                    if types[col] == 'object' or nunique[col] < 100:
                        categorical_indicator[X.columns.get_loc(col)] = True

                categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

                cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                X_num_all = df.iloc[:, con_idxs].astype(np.float32).values
                X_cat_all = df.iloc[:, cat_idxs].astype(str).values
                if (len(train_index) == 0):
                    X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
                    train_index = X[X.Set == "train"].index
                    valid_index = X[X.Set == "valid"].index
                    test_index = X[X.Set == "test"].index
                    X = X.drop(columns=['Set'])
                X = X.to_numpy()
                X_num = {}
                X_cat = {}
                y = {}
                combined_array = np.hstack((X[train_index], y_all[train_index].reshape(-1, 1)))
                # 根据最后一列是否为空来过滤样本
                train_non_empty_rows = combined_array[~np.isnan(combined_array[:, -1])]
                X_num["train"] = train_non_empty_rows[:, con_idxs]
                X_cat["train"] = train_non_empty_rows[:, cat_idxs]
                y["train"] = train_non_empty_rows[:, -1]
                combined_array = np.hstack((X[valid_index], y_all[valid_index].reshape(-1, 1)))
                # 根据最后一列是否为空来过滤样本
                val_non_empty_rows = combined_array[~np.isnan(combined_array[:, -1])]
                X_num["val"] = val_non_empty_rows[:, con_idxs]
                X_cat["val"] = val_non_empty_rows[:, cat_idxs]
                y["val"] = val_non_empty_rows[:, -1]
                X_num["test"] = X_num_all[test_index]
                X_cat["test"] = X_cat_all[test_index]
                y["test"] = y_all[test_index]
                # X_num_all, y_all = _get_sklearn_dataset('california_housing')
                # idx = _make_split(len(X_num_all), None, 3)

                _save_label(
                    dataset_dir,
                    'News',
                    TaskType.BINCLASS,
                    X_num = X_num,
                    X_cat = X_cat,
                    y = y,
                )
def temperature_label():
    # dataset_dir, _ = _start('HI')

    # dataset_dir = "data/HI"
    miss_types = ["mcar_label_"]
    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    imputations = ["_missforest"]
    train_index = []
    valid_index = []
    test_index = []
    for rate in missing_rates:
        for miss_type in miss_types:
            for imp in imputations:
                _set_random_seeds()
                if imp != "mean":
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/temperature/" + miss_type + imp + str(
                            rate))
                    out = Path("/data/lsw/data/data/temperature/" + miss_type + "temperature_" + str(
                        rate) + imp + ".csv")
                else:
                    dataset_dir = Path(
                        "/data/lsw/tabular/tabular-dl-num-embeddings-main-our/data/temperature/" + miss_type + str(
                            rate))
                    out = Path("/data/lsw/data/data/temperature/" + miss_type + "temperature_" + str(
                        rate) + ".csv")
                if not os.path.exists(dataset_dir):
                    os.mkdir(dataset_dir)
                df = pd.read_csv(out)
                y_all = df.pop('target').values  # type: ignore[code]
                X = df
                nunique = X.nunique()
                types = X.dtypes
                categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
                for col in X.columns:
                    if types[col] == 'object' or nunique[col] < 100:
                        categorical_indicator[X.columns.get_loc(col)] = True

                categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

                cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                X_num_all = df.iloc[:, con_idxs].astype(np.float32).values
                X_cat_all = df.iloc[:, cat_idxs].astype(str).values
                if (len(train_index) == 0):
                    X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
                    train_index = X[X.Set == "train"].index
                    valid_index = X[X.Set == "valid"].index
                    test_index = X[X.Set == "test"].index
                    X = X.drop(columns=['Set'])
                X = X.to_numpy()
                X_num = {}
                X_cat = {}
                y = {}
                try:
                    X = X.values()
                except:
                    X = X
                combined_array = np.hstack((X[train_index], y_all[train_index].reshape(-1, 1)))
                # 根据最后一列是否为空来过滤样本
                train_non_empty_rows = combined_array[~np.isnan(combined_array[:, -1])]
                X_num["train"] = train_non_empty_rows[:, con_idxs]
                X_cat["train"] = train_non_empty_rows[:, cat_idxs]
                y["train"] = train_non_empty_rows[:, -1]
                combined_array = np.hstack((X[valid_index], y_all[valid_index].reshape(-1, 1)))
                # 根据最后一列是否为空来过滤样本
                val_non_empty_rows = combined_array[~np.isnan(combined_array[:, -1])]
                X_num["val"] = val_non_empty_rows[:, con_idxs]
                X_cat["val"] = val_non_empty_rows[:, cat_idxs]
                y["val"] = val_non_empty_rows[:, -1]
                X_num["test"] = X_num_all[test_index]
                X_cat["test"] = X_cat_all[test_index]
                y["test"] = y_all[test_index]
                # X_num_all, y_all = _get_sklearn_dataset('california_housing')
                # idx = _make_split(len(X_num_all), None, 3)

                _save_label(
                    dataset_dir,
                    'temperature',
                    TaskType.REGRESSION,
                    X_num = X_num,
                    X_cat = X_cat,
                    y = y,
                )
def main(argv):
    _set_random_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Remove everything except for the expected files.',
    )
    args = parser.parse_args(argv[1:])
    News_label()
    temperature_label()
    print('-----')
    print('Done!')


if __name__ == '__main__':
    sys.exit(main(sys.argv))