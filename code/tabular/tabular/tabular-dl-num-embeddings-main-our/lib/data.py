import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import env, util
from .metrics import calculate_metrics as calculate_metrics_
from .util import TaskType

ArrayDict = dict[str, np.ndarray]
TensorDict = dict[str, torch.Tensor]


CAT_MISSING_VALUE = np.nan
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']


class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def get_category_sizes(X: Union[torch.Tensor, np.ndarray], val_X: Union[torch.Tensor, np.ndarray], test_X: Union[torch.Tensor, np.ndarray]) -> list[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    XV = val_X.T.cpu().tolist() if isinstance(val_X, torch.Tensor) else val_X.T.tolist()
    XTE = test_X.T.cpu().tolist() if isinstance(test_X, torch.Tensor) else test_X.T.tolist()
    merged_list = []
    for tuple_item in list(zip(XT, XV, XTE)):
        merged_sublist = []
        for sub_list in tuple_item:
            merged_sublist.extend(sub_list)
        merged_list.append(merged_sublist)
    XT = merged_list
    return [len(set(x)) for x in XT]

def generate_new_matrix(A, percentage_to_set_ones=0.05):
    A = np.array(A)
    # Step 1: Create a zero-filled matrix B with the same shape as A
    B = np.zeros_like(A).astype(bool)

    # Step 2: Set positions with 1 in A to 0 in B
    # Step 3: Set 10% of the positions with 0 in A to 1 in B
    num_zeros = np.sum(A == False)
    num_values_to_set_one = int(num_zeros * percentage_to_set_ones)

    # Generate random indices to set to 1
    random_indices_to_set_one = np.random.choice(num_zeros, num_values_to_set_one, replace=False)
    zero_indices = np.where(A == False)
    # Set the chosen positions to 1 in B
    B[zero_indices[0][random_indices_to_set_one], zero_indices[1][random_indices_to_set_one]] = True
    return B

@dataclass()
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]
    mask_metrics: Optional[list]
    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> 'Dataset':
        dir_ = Path(dir_)

        def load(item) -> ArrayDict:
            # train: .npy test: .npy val: .npy
            return {
                x: cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy'))  # type: ignore[code]
                for x in ['train', 'val', 'test']
            }

        info = util.load_json(dir_ / 'info.json')

        return Dataset(
            load('X_num') if dir_.joinpath('X_num_train.npy').exists() else None,
            load('X_cat') if dir_.joinpath('X_cat_train.npy').exists() else None,
            load('y'),
            {},
            # 用于分配损失函数类型
            TaskType(info['task_type']),
            info.get('n_classes'),
            [],
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1
    @property
    def categories(self):
        concatenated_arr = np.concatenate((self.X_cat['train'], self.X_cat['val'], self.X_cat['test']), axis=0)
        unique_counts = np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=concatenated_arr)
        return unique_counts

    def get_category_sizes(self, part: str) -> list[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part], self.X_cat['val'], self.X_cat['test'])

    def set_mask_metric(self, num_mask_dict, cat_mask_dict, mask_rate):
        mask_metric = [{}, {}]
        if self.X_num:
            mask_metrics = torch.tensor(generate_new_matrix(num_mask_dict['train'], mask_rate))

            mask_metric[0]['train'] = mask_metrics
            self.X_num['train'][mask_metrics] = 0
        if self.X_cat:
            mask_metrics = torch.tensor(generate_new_matrix(cat_mask_dict['train'], mask_rate))
            mask_metric[1]['train'] = mask_metrics
            self.X_cat['train'][mask_metrics] = 0
        self.mask_metrics = mask_metric
    def calculate_metrics(
        self,
        predictions: dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> dict[str, Any]:
        metrics = {
            x: calculate_metrics_(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = 'rmse'
            score_key_2 = 'R2'
            score_sign = -1
        else:
            score_key = 'accuracy'
            score_key_2 = 'roc_auc'
            score_sign = 1
            
        for part_metrics in metrics.values():
            part_metrics['score'] = score_sign * part_metrics[score_key]
            part_metrics['score2'] = part_metrics[score_key_2]
        return metrics

def generate_zero_one_matrix(shape, percentage=0.1):
    # Step 1: Create a zero-filled matrix with the given shape
    zero_matrix = np.zeros(shape)

    # Step 2: Set 10% of the positions to 1
    total_elements = np.prod(shape)  # Calculate the total number of elements in the matrix
    num_ones = int(percentage * total_elements)  # Calculate the number of 1's to be set

    # Generate random indices to set to 1
    random_indices = np.random.choice(total_elements, num_ones, replace=False)
    # Convert flat indices to 2D indices
    row_indices, col_indices = np.unravel_index(random_indices, shape)
    # Set the chosen positions to 1
    zero_matrix[row_indices, col_indices] = 1

    return zero_matrix

def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        assert policy is None
        return dataset
    policy = 'mean'
    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        mask_metric = {}
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert util.raise_unknown('policy', policy)
    return dataset

# Inspired by: https://github.com/Yura52/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X: ArrayDict, normalization: Normalization, seed: Optional[int]
) -> ArrayDict:
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
        noise = 1e-3
        if noise > 0:
            assert seed is not None
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
                X_train.shape
            )
    else:
        util.raise_unknown('normalization', normalization)
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}  # type: ignore[code]


def cat_process_nans(dataset, X: ArrayDict, policy: Optional[CatNanPolicy], nan_masks) -> ArrayDict:
    assert X is not None
    X['train'] = X['train'].astype(float)
    X['val'] = X['val'].astype(float)
    X['test'] = X['test'].astype(float)
    # nan_masks = {k: v.any() == CAT_MISSING_VALUE for k, v in X.items()}
    policy = 'most_frequent'
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            util.raise_unknown('categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
) -> tuple[ArrayDict, bool]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # Step 1. Map strings to 0-based ranges
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype='int64',  # type: ignore[code]
    ).fit(X['train'])
    X = {k: encoder.transform(v) for k, v in X.items()}
    max_values = X['train'].max(axis=0)
    for part in ['val', 'test']:
        for column_idx in range(X[part].shape[1]):
            X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    # Step 2. Encode.
    if encoding is None:
        return (X, False)
    elif encoding == 'one-hot':
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32  # type: ignore[code]
        )
        encoder.fit(X['train'])
        return ({k: encoder.transform(v) for k, v in X.items()}, True)  # type: ignore[code]
    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        encoder = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.fit(X['train'], y_train)
        X = {k: encoder.transform(v).astype('float32') for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X['train'], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
        return (X, True)  # type: ignore[code]
    else:
        util.raise_unknown('encoding', encoding)


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> tuple[ArrayDict, dict[str, Any]]:
    info: dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            # y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    else:
        util.raise_unknown('policy', policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
    nan_mask: Optional
) -> Dataset:
    # 在build_dataset中调用

    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    cache_dir = None
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f'cache__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = util.load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    X_num = dataset.X_num
    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset, dataset.X_cat, transformations.cat_nan_policy, nan_mask)

        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y['train'],
            transformations.seed,
        )
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    if X_num is not None and transformations.normalization is not None:
        X_num = normalize(X_num, transformations.normalization, transformations.seed)

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    if cache_path is not None:
        util.dump_pickle((transformations, dataset), cache_path)
    return dataset

# def build_dataset(
#     path: Union[str, Path], transformations: Transformations, cache: bool, num_mask_dict, cat_mask_dict, mask_rate
# ) -> Dataset:
#     path = Path(path)
#     dataset = Dataset.from_dir(path)
#     dataset.set_mask_metric(num_mask_dict, cat_mask_dict, mask_rate)
#     return transform_dataset(dataset, transformations, path if cache else None, cat_mask_dict)
def build_dataset(
    path: Union[str, Path], transformations: Transformations, cache: bool,  cat_mask_dict
) -> Dataset:
    path = Path(path)
    
    # 此处需要使用info.json文件
    dataset = Dataset.from_dir(path)

    # 此处与y_info有关
    return transform_dataset(dataset, transformations, path if cache else None, cat_mask_dict)


def prepare_tensors(
    dataset: Dataset, device: Union[str, torch.device]
) -> tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    if isinstance(device, str):
        device = torch.device(device)
    # 原
    # X_num, X_cat, Y = (
    #     None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
    #     for x in [dataset.X_num, dataset.X_cat, dataset.y]
    # )
        
    """
    lzy begin
    转换dtype
    """
    X_num = None if dataset.X_num is None else {k: torch.as_tensor(v, dtype=torch.float32) for k, v in dataset.X_num.items()}
    X_cat = None if dataset.X_cat is None else {k: torch.as_tensor(v, dtype=torch.int64) for k, v in dataset.X_cat.items()}
    
    if dataset.is_regression:
        Y = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in dataset.y.items()}
    else:
        Y = {k: torch.as_tensor(v, dtype=torch.int64) for k, v in dataset.y.items()}
    
    """
    lzy end
    """
    
    if device.type != 'cpu':
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    return X_num, X_cat, Y


def load_dataset_info(dataset_dir_name: str) -> dict[str, Any]:
    path = env.DATA / dataset_dir_name
    info = util.load_json(path / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    info['n_features'] = info['n_num_features'] + info['n_cat_features']
    info['path'] = path
    return info
