# %%
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union, cast
import random
from pathlib import Path
import pandas as pd
import numpy as np
import rtdl_our
import torch
import torch.nn as nn
import zero
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch import Tensor
from torch.nn import Parameter  # type: ignore[code]
from tqdm import trange
import lib
import json
from IPS import sampling, compute_ips
import torch.optim as optim

# added by lzy 7.31
import os
# %%
@dataclass
class FourierFeaturesOptions:
    n: int  # the output size is 2 * n
    sigma: float


@dataclass
class AutoDisOptions:
    n_meta_embeddings: int
    temperature: float


@dataclass
class Config:
    @dataclass
    class Data:
        dataset: str
        type: str
        missingrate: float
        path: str
        ipsnum: int
        T: lib.Transformations = field(default_factory=lib.Transformations)
        T_cache: bool = False

    @dataclass
    class Bins:
        count: int
        encoding: Literal['piecewise-linear', 'binary', 'one-blob'] = 'piecewise-linear'
        one_blob_gamma: Optional[float] = None
        tree: Optional[dict[str, Any]] = None
        subsample: Union[None, int, float] = None

    @dataclass
    class Model:
        d_num_embedding: Optional[int] = None
        num_embedding_arch: list[str] = field(default_factory=list)
        d_cat_embedding: Union[None, int, Literal['d_num_embedding']] = None
        mlp: Optional[dict[str, Any]] = None
        resnet: Optional[dict[str, Any]] = None
        transformer: Optional[dict[str, Any]] = None
        transformer_default: bool = False
        transformer_baseline: bool = True
        periodic_sigma: Optional[float] = None
        periodic: Optional[lib.PeriodicOptions] = None
        autodis: Optional[AutoDisOptions] = None
        dice: bool = False
        fourier_features: Optional[FourierFeaturesOptions] = None
        # The following parameter is purely technical and does not affect the "algorithm".
        # Setting it to False leads to better speed.
        memory_efficient: bool = False

    @dataclass
    class Training:
        batch_size: int
        lr: float
        weight_decay: float
        optimizer: str = 'AdamW'
        patience: Optional[int] = 16
        n_epochs: Union[int, float] = math.inf
        eval_batch_size: int = 8192

    seed: int
    data: Data
    model: Model
    training: Training
    bins: Optional[Bins] = None

    @property
    def is_mlp(self):
        return self.model.mlp is not None

    @property
    def is_resnet(self):
        return self.model.resnet is not None

    @property
    def is_transformer(self):
        return self.model.transformer is not None

    def __post_init__(self):
        assert sum([self.is_mlp, self.is_resnet, self.is_transformer]) == 1
        if self.bins is not None and self.bins.encoding == 'one-blob':
            assert self.bins.one_blob_gamma is not None
        if self.model.periodic_sigma is not None:
            assert self.model.periodic is None
            assert self.model.d_num_embedding is not None
            assert self.model.d_num_embedding % 2 == 0
            self.model.periodic = lib.PeriodicOptions(
                self.model.d_num_embedding // 2,
                self.model.periodic_sigma,
                False,
                'log-linear',
            )
            self.model.periodic_sigma = None
            if self.model.num_embedding_arch == ['positional']:
                self.model.d_num_embedding = None
        if self.model.periodic is not None:
            assert self.model.fourier_features is None
        if self.model.d_num_embedding is not None:
            assert self.model.num_embedding_arch or self.model.dice
        if self.model.dice:
            assert self.model.d_num_embedding
        if self.is_resnet:
            lib.replace_factor_with_value(
                self.model.resnet,  # type: ignore[code]
                'd_hidden',
                self.model.resnet['d_main'],  # type: ignore[code]
                (1.0, 8.0),
            )
        if self.is_transformer and not self.model.transformer_default:
            assert self.model.d_num_embedding is not None
            assert self.model.fourier_features is None
            lib.replace_factor_with_value(
                self.model.transformer,
                'ffn_d_hidden',
                self.model.d_num_embedding,
                (0.5, 4.0),
            )


class NLinear(nn.Module):
    def __init__(self, n: int, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n, d_in, d_out))
        self.bias = Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class NLinearMemoryEfficient(nn.Module):
    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class NLayerNorm(nn.Module):
    def __init__(self, n_features: int, d: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.ones(n_features, d))
        self.bias = Parameter(torch.zeros(n_features, d))

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        x = self.weight * x + self.bias
        return x


class AutoDis(nn.Module):
    """
    Paper (the version is important): https://arxiv.org/abs/2012.08986v2
    Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis

    The paper is significantly different from the code (it looks like the code
    implements the first version of the paper). We implement the second version
    here. Not all technical details are given for the second version, so what we do
    here can be different from what authors actually did.

    Anyway, AutoDis (v2) is essentially the following sequence of layers (applied from
    left to right): [Linear(no bias), LeakyReLU, Linear(no bias), Softmax, Linear]
    """

    def __init__(
        self, n_features: int, d_embedding: int, options: AutoDisOptions
    ) -> None:
        super().__init__()
        self.first_layer = rtdl_our.NumericalFeatureTokenizer(
            n_features,
            options.n_meta_embeddings,
            False,
            'uniform',
        )
        self.leaky_relu = nn.LeakyReLU()
        self.second_layer = NLinear(
            n_features, options.n_meta_embeddings, options.n_meta_embeddings, False
        )
        self.softmax = nn.Softmax(-1)
        self.temperature = options.temperature
        # "meta embeddings" from the paper are just a linear layer
        self.third_layer = NLinear(
            n_features, options.n_meta_embeddings, d_embedding, False
        )
        # 0.01 is taken from the source code
        nn.init.uniform_(self.third_layer.weight, 0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.leaky_relu(x)
        x = self.second_layer(x)
        x = self.softmax(x / self.temperature)
        x = self.third_layer(x)
        return x


class NumEmbeddings(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_embedding: Optional[int],
        embedding_arch: list[str],
        periodic_options: Optional[lib.PeriodicOptions],
        autodis_options: Optional[AutoDisOptions],
        d_feature: Optional[int],
        memory_efficient: bool,
    ) -> None:
        super().__init__()
        assert embedding_arch
        assert set(embedding_arch) <= {
            'linear',
            'positional',
            'relu',
            'shared_linear',
            'layernorm',
            'autodis',
        }
        if any(x in embedding_arch for x in ['linear', 'shared_linear', 'autodis']):
            assert d_embedding is not None
        else:
            assert d_embedding is None
        assert embedding_arch.count('positional') <= 1
        if 'autodis' in embedding_arch:
            assert embedding_arch == ['autodis']

        NLinear_ = NLinearMemoryEfficient if memory_efficient else NLinear
        layers: list[nn.Module] = []

        if embedding_arch[0] == 'linear':
            assert periodic_options is None
            assert autodis_options is None
            assert d_embedding is not None
            layers.append(
                rtdl_our.NumericalFeatureTokenizer(n_features, d_embedding, True, 'uniform')
                if d_feature is None
                else NLinear_(n_features, d_feature, d_embedding)
            )
            d_current = d_embedding
        elif embedding_arch[0] == 'positional':
            assert d_feature is None
            assert periodic_options is not None
            assert autodis_options is None
            layers.append(lib.Periodic(n_features, periodic_options))
            d_current = periodic_options.n * 2
        elif embedding_arch[0] == 'autodis':
            assert d_feature is None
            assert periodic_options is None
            assert autodis_options is not None
            assert d_embedding is not None
            layers.append(AutoDis(n_features, d_embedding, autodis_options))
            d_current = d_embedding
        else:
            assert False

        for x in embedding_arch[1:]:
            layers.append(
                nn.ReLU()
                if x == 'relu'
                else NLinear_(n_features, d_current, d_embedding)  # type: ignore[code]
                if x == 'linear'
                else nn.Linear(d_current, d_embedding)  # type: ignore[code]
                if x == 'shared_linear'
                else NLayerNorm(n_features, d_current)  # type: ignore[code]
                if x == 'layernorm'
                else nn.Identity()
            )
            if x in ['linear', 'shared_linear']:
                d_current = d_embedding
            assert not isinstance(layers[-1], nn.Identity)
        self.d_embedding = d_current
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _DICE(nn.Module):
    """The DICE method from "Methods for Numeracy-Preserving Word Embeddings" by Sundararaman et al."""

    Q: Tensor

    def __init__(self, d: int, x_min: float, x_max: float) -> None:
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        M = torch.randn(d, d)
        Q, _ = torch.linalg.qr(M)
        self.register_buffer('Q', Q)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 1
        d = len(self.Q)
        x = (x - self.x_min) / (self.x_max - self.x_min)
        x = torch.where(
            (0.0 <= x) & (x <= 1.0),
            x * torch.pi,
            torch.empty_like(x).uniform_(-torch.pi, torch.pi)
            # torch.distributions.Uniform(-torch.pi, torch.pi).sample(x.shape).to(x),
        )
        exponents = torch.arange(d - 1, dtype=x.dtype, device=x.device)
        x = torch.column_stack(
            [
                torch.cos(x)[:, None] * torch.sin(x)[:, None] ** exponents[None],
                torch.sin(x) ** (d - 1),
            ]
        )
        x = x @ self.Q
        return x


class DICEEmbeddings(nn.Module):
    def __init__(
        self, d: int, lower_bounds: list[float], upper_bounds: list[float]
    ) -> None:
        super().__init__()
        self.modules_ = nn.ModuleList(
            [_DICE(d, *bounds) for bounds in zip(lower_bounds, upper_bounds)]
        )
        self.d_embedding = d

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == len(self.modules_)
        return torch.stack(
            [self.modules_[i](x[:, i]) for i in range(len(self.modules_))],
            1,
        )


class BaseModel(nn.Module):
    category_sizes: List[int]  # torch.jit does not support list[int]

    def __init__(self, config: Config, dataset: lib.Dataset, n_bins: Optional[int]):
        super().__init__()
        assert dataset.X_num is not None
        lower_bounds = dataset.X_num['train'].min().tolist()
        upper_bounds = dataset.X_num['train'].max().tolist()
        self.num_embeddings = (
            NumEmbeddings(
                D.n_num_features,
                config.model.d_num_embedding,
                config.model.num_embedding_arch,
                config.model.periodic,
                config.model.autodis,
                n_bins,
                config.model.memory_efficient,
            )
            if config.model.num_embedding_arch
            else DICEEmbeddings(
                cast(int, config.model.d_num_embedding), lower_bounds, upper_bounds
            )
            if config.model.dice
            else None
        )
        self.category_sizes = dataset.get_category_sizes('train')
        d_cat_embedding = (
            config.model.d_num_embedding
            if self.category_sizes
            and (
                config.is_transformer
                or config.model.d_cat_embedding == 'd_num_embedding'
            )
            else config.model.d_cat_embedding
        )
        assert d_cat_embedding is None or isinstance(d_cat_embedding, int)
        self.cat_embeddings = (
            None
            if d_cat_embedding is None
            else rtdl_our.CategoricalFeatureTokenizer(
                self.category_sizes, d_cat_embedding, True, 'uniform'
            )
        )

    def _encode_input(
        self, x_num: Optional[Tensor], x_cat: Optional[Tensor]
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if self.num_embeddings is not None:
            assert x_num is not None
            t = torch.isnan(x_num)
            if(torch.any(t)):
                print("youqdhqo")
            x_num = self.num_embeddings(x_num)
        if self.cat_embeddings is not None:
            assert x_cat is not None
            t = torch.isnan(x_cat)
            if (torch.any(t)):
                print("youqdhqo")
            x_cat = self.cat_embeddings(x_cat)
        elif x_cat is not None:
            x_cat = torch.concat(
                [
                    nn.functional.one_hot(x_cat[:, i], category_size)  # type: ignore[code]
                    for i, category_size in enumerate(self.category_sizes)
                ],
                1,
            )
        return x_num, x_cat


class FlatModel(BaseModel):
    def __init__(self, config: Config, dataset, n_bins: Optional[int]):
        super().__init__(config, dataset, n_bins)
        # 数值特征的输入维度
        d_num_embedding = (
            None if self.num_embeddings is None else self.num_embeddings.d_embedding
        )

        # 总输入维度 特征数量 × 每个特征的维度(嵌入使用d_num_embedding, 分箱使用n_bins， 标量使用1)
        d_num_in = D.n_num_features * (d_num_embedding or n_bins or 1)

        # 一种特征变换 将原始数值特征映射到高维的三角函数空间
        if config.model.fourier_features is not None:
            assert self.num_embeddings is None
            assert d_num_in
            self.fourier_matrix = nn.Linear(
                d_num_in, config.model.fourier_features.n, False
            )

            # 正态分布初始化权重
            nn.init.normal_(
                self.fourier_matrix.weight,
                std=config.model.fourier_features.sigma ** 2,
            )
            
            self.fourier_matrix.weight.requires_grad_(False)
            d_num_in = config.model.fourier_features.n * 2
        else:
            self.fourier_matrix = None
        
        # 分类特征的输入维度
        d_cat_in = (
            sum(self.category_sizes)
            if config.model.d_cat_embedding is None
            else len(self.category_sizes) * config.model.d_cat_embedding
        )
        assert isinstance(d_cat_in, int)

        # 配置主干网络
        in_out_options = {
            'd_in': d_num_in + d_cat_in,
            'd_out': dataset.nn_output_dim,
        }

        # 选择主干网络
        if config.is_mlp:
            self.main_module = rtdl_our.MLP.make_baseline(
                **config.model.mlp, **in_out_options  # type: ignore[code]
            )
        elif config.is_resnet:
            self.main_module = rtdl_our.ResNet.make_baseline(
                **config.model.resnet, **in_out_options  # type: ignore[code]
            )
        else:
            assert False

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        assert x_num is not None or x_cat is not None
        x = self._encode_input(x_num, x_cat)
        x = [
            None if x_ is None else x_.flatten(1, 2) if x_.ndim == 3 else x_ for x_ in x
        ]
        if self.fourier_matrix is not None:
            assert x[0] is not None
            x[0] = lib.cos_sin(2 * torch.pi * self.fourier_matrix(x[0]))

        # torch.jit does not support list comprehensions with conditions
        x_tensors = []
        for x_ in x:
            if x_ is not None:
                x_tensors.append(x_)
        x = torch.concat(x_tensors, 1)
        return self.main_module(x)

class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
# main class

class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred

class NonFlatModel(BaseModel):
    def __init__(self, config: Config, dataset, n_bins: Optional[int]):
        super().__init__(config, dataset, n_bins)
        assert config.model.transformer is not None
        assert self.num_embeddings is not None
        
        transformer_options = deepcopy(config.model.transformer)
        if config.model.transformer_default:
            transformer_options = (
                rtdl_our.FTTransformer.get_default_transformer_config(
                    n_blocks=transformer_options.get('n_blocks', 3)
                )
                | transformer_options
            )
        elif config.model.transformer_baseline:
            transformer_options = (
                rtdl_our.FTTransformer.get_baseline_transformer_subconfig()
                | transformer_options
            )
        d_cat_embedding = (
            None if self.cat_embeddings is None else self.cat_embeddings.d_token
        )
        d_embedding = config.model.d_num_embedding or d_cat_embedding
        assert d_embedding is not None
        self.cls_embedding = rtdl_our.CLSToken(d_embedding, 'uniform')
        self.main_module = rtdl_our.Transformer(
            d_token=d_embedding,
            **transformer_options,
            d_out=dataset.nn_output_dim,
        )
        self.num_continuous = dataset.n_num_features
        self.num_categories = dataset.n_cat_features
        self.pt_mlp = simple_MLP(
            [d_embedding * (self.num_continuous + self.num_categories),
             6 * d_embedding * (self.num_continuous + self.num_categories) // 5,
             d_embedding * (self.num_continuous + self.num_categories) // 2])
        if self.num_categories != 0:
            self.mlp1 = sep_MLP(d_embedding, self.num_categories, dataset.categories)
        self.mlp2 = sep_MLP(d_embedding, self.num_continuous, np.ones(self.num_continuous).astype(int))
    def forward(self, x_num, x_cat, num_mask, cat_mask, num_ips, cat_ips):
        assert x_num is not None or x_cat is not None
        x = self._encode_input(x_num, x_cat)
        for x_ in x:
            if x_ is not None:
                assert x_.ndim == 3
        ips = torch.concat([num_ips, cat_ips], 1)
        mask = torch.concat([num_mask, cat_mask], 1)
        x = torch.concat([x_ for x_ in x if x_ is not None], 1)
        x = self.cls_embedding(x)
        return self.main_module(x, ips, mask)


# %%
def patch_raw_config(raw_config):
    # Before this option was introduced, the code was always "memory efficient"
    raw_config['model'].setdefault('memory_efficient', True)

    bins = raw_config.get('bins')
    if bins is not None:
        if 'encoding' in bins:
            assert 'value' not in bins
        elif 'value' in bins:
            value = bins.pop('value')
            if value == 'one':
                bins['encoding'] = 'binary'
            elif value == 'ratio':
                bins['encoding'] = 'piecewise-linear'
            else:
                assert False
        else:
            bins['encoding'] = 'piecewise-linear'
        if bins['encoding'] == 'one':
            bins['encoding'] = 'binary'

    key = 'positional_encoding'
    if key in raw_config['model']:
        print(
            f'WARNING: "{key}" is a deprecated alias for "periodic", use "periodic" instead'
        )
        raw_config['model']['periodic'] = raw_config['model'].pop(key)
        time.sleep(1.0)



def encode(part, idx):
    assert C.bins is not None
    assert bins is not None
    assert n_bins is not None

    if C.bins.encoding == 'one-blob':
        assert bin_edges is not None
        assert X_num is not None
        assert C.bins.one_blob_gamma is not None
        x = torch.zeros(
            len(idx), D.n_num_features, n_bins, dtype=torch.float32, device=device
        )
        for i in range(D.n_num_features):
            n_bins_i = len(bin_edges[i]) - 1
            bin_left_edges = bin_edges[i][:-1]
            bin_right_edges = bin_edges[i][1:]
            kernel_scale = 1 / (n_bins_i ** C.bins.one_blob_gamma)
            cdf_values = [
                0.5
                * (
                    1
                    + torch.erf(
                        (edges[None] - X_num[part][idx, i][:, None])
                        / (kernel_scale * 2 ** 0.5)
                    )
                )
                for edges in [bin_left_edges, bin_right_edges]
            ]
            x[:, i, :n_bins_i] = cdf_values[1] - cdf_values[0]

    else:
        assert bin_values is not None
        bins_ = bins[part][idx]
        bin_mask_ = torch.eye(n_bins, device=device)[bins_]
        x = bin_mask_ * bin_values[part][idx, ..., None]
        previous_bins_mask = torch.arange(n_bins, device=device)[None, None].repeat(
            len(idx), D.n_num_features, 1
        ) < bins_.reshape(len(idx), D.n_num_features, 1)
        x[previous_bins_mask] = 1.0


    return x





def apply_model(part, idx):
    return model(
        (
            encode(part, idx)
            if C.bins is not None
            else X_num[part][idx]
            if X_num is not None
            else None
        ),
        None if X_cat is None else X_cat[part][idx],
        num_nan_mask_dict[part][idx],
        cat_nan_mask_dict[part][idx],
        num_ips_dict[part][idx],
        cat_ips_dict[part][idx]
    ).squeeze(-1)




def print_epoch_info():
    print(f'\n>>> Epoch {stream.epoch} | {timer} | {output}')
    print(
        ' | '.join(
            f'{k} = {v}'
            for k, v in {
                'lr': lib.get_lr(optimizer),
                'batch_size': C.training.batch_size,  # type: ignore[code]
                'chunk_size': chunk_size,
                'epoch_size': report['epoch_size'],
                'n_parameters': report['n_parameters'],
            }.items()
        )
    )


def are_valid_predictions(predictions: dict[str, np.ndarray]) -> bool:
    return all(np.isfinite(x).all() for x in predictions.values())


@torch.inference_mode()
def evaluate(parts):
    global eval_batch_size
    model.eval()
    predictions = {}
    for part in parts:
        while eval_batch_size:
            try:
                predictions[part] = (
                    torch.cat(
                        [
                            apply_model(part, idx)
                            for idx in zero.data.IndexLoader(
                                D.size(part), eval_batch_size, False, device=device
                            )
                        ]
                    )
                    .cpu()
                    .numpy()
                )
            except RuntimeError as err:
                if not lib.is_oom_exception(err):
                    raise
                eval_batch_size //= 2
                print('New eval batch size:', eval_batch_size)
                report['eval_batch_size'] = eval_batch_size
            else:
                break
        if not eval_batch_size:
            RuntimeError('Not enough memory even for eval_batch_size=1')
    metrics = (
        D.calculate_metrics(predictions, report['prediction_type'])
        if are_valid_predictions(predictions)
        else {x: {'score': -999999.0, 'score2': -99999.0} for x in predictions}
    )
    return metrics, predictions


def save_checkpoint():
    torch.save(
        {
            'model': model.state_dict(),
        },
        checkpoint_path,
    )
    # lib.dump_report(report, output)
    # lib.backup_output(output)

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
criterion3 = nn.LogSoftmax(dim=1)

def pretrain_module(model, x: Tensor, ips, mask):
    # 遍历每一层
    for layer_idx, layer in enumerate(model.main_module.blocks):
        layer = cast(nn.ModuleDict, layer)

        query_idx = (
            model.main_module.last_layer_query_idx if layer_idx + 1 == len(model.main_module.blocks) else None
        )
        x_residual = model.main_module._start_residual(layer, 'attention', x)
        x_residual, _ = layer['attention'](
            x_residual if query_idx is None else x_residual[:, query_idx],
            x_residual,
            ips,
            mask,
            *model.main_module._get_kv_compressions(layer),
        )

        # 根据query_index选择Q
        if query_idx is not None:
            x = x[:, query_idx]
        x = model.main_module._end_residual(layer, 'attention', x, x_residual)

        x_residual = model.main_module._start_residual(layer, 'ffn', x)
        x_residual = layer['ffn'](x_residual)
        x = model.main_module._end_residual(layer, 'ffn', x, x_residual)
        x = layer['output'](x)
    return x


missingrates = [0.1, 0.3, 0.5, 0.7, 0.9]
datasets = ["HI", "News", "temperature", "gas"]
missingtypes = ["mcar_", "mar_p_", "mnar_p_"]
# missingtypes = ["mnar_p_"]
ips_nums = [40]
seeds = [0, 149669, 52983]
imputations = ["mean"]
# missingrates = [0.5]
# bi_datasets = ["News"]
# mul_datasets = []
# reg_datasets = []
# missingtypes = ["mcar_"]
# ips_nums = [40]
# seeds = [0]
all_result = []
t = time.localtime()
t_time = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + str(t.tm_min)
if __name__ == "__main__":
    # datasets = ["HI", "News", "temperature", "gas"]
    for data_name in datasets:
        # missingrates = [0.1, 0.3, 0.5, 0.7, 0.9]
        for missingrate in missingrates:
            # missingtypes = ["mcar_", "mar_p_", "mnar_p_"]
            for missingtype in missingtypes:
                # imputations = ["mean"]
                for imp in imputations:
                    # ips_nums = [40]
                    for ips_num in ips_nums:
                        # seeds = [0, 149669, 52983]
                        for seed in seeds:
                            start_time = time.time()

                            # 读取.toml配置文件 完成对C所有属性的赋值 可能影响out
                            if missingtype == "mnar_p_":
                                config_path = "exp/transformer-q-lr/" + data_name + "/" + "mnar_" + str(
                                    0.5) + ".toml"
                            elif missingtype == "mar_p_":
                                config_path = "exp/transformer-q-lr/" + data_name + "/" + "mar_" + str(
                                    0.5) + ".toml"
                            else:
                                config_path = "exp/transformer-q-lr/" + data_name + "/" + "mcar_" + str(
                                    0.5) + ".toml"
                                
                            C, output, report = lib.start(Config, patch_raw_config=patch_raw_config,
                                                          config_path=config_path)
                            
                            if imp != "mean":
                                C.data.path = "data/" + data_name + "/" + missingtype + imp + str(missingrate)
                            else:
                                C.data.path = "data/" + data_name + "/" + missingtype + str(missingrate)

                            """
                            added by lzy 7.31
                            sh scripts/tabular.sh
                            调试工作路径

                            output:
                            当前工作目录: /data/lzy/MISS2
                            相对路径: data/HI/mcar_0.1
                            绝对路径: /data/lzy/MISS2/data/HI/mcar_0.1
                            """
                            # print(f"当前工作目录: {os.getcwd()}")
                            # print(f"相对路径: {C.data.path}")
                            # print(f"绝对路径: {Path(C.data.path).resolve()}")

                            # out = Path(
                            #     "/data/lsw/data/data/" + C.data.dataset + "/" + C.data.type + C.data.dataset + "_" + str(
                            #         C.data.missingrate) + ".csv")

                            # 数据集路径
                            out = Path(
                                "/home/lzy/data/MISS2/dataset/" + C.data.dataset + "/" + C.data.type + C.data.dataset + "_" + str(
                                    C.data.missingrate) + ".csv")
                            
                            random.seed(0)
                            np.random.seed(0)
                            
                            # 获取设备
                            zero.improve_reproducibility(seed)
                            device = lib.get_device()

                            # 读入数据
                            data = pd.read_csv(out)
                            X = data.iloc[:, :-1]

                            nan_mask = pd.isna(X).to_numpy() # 缺失值mask
                            nan_mask = torch.tensor(nan_mask, device=device)  # 添加device参数
                            X = pd.DataFrame(X)

                            nunique = X.nunique()

                            types = X.dtypes

                            categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))  # 识别每列是否为离散特征
                            for col in X.columns:
                                if types[col] == 'object' or nunique[col] < 100:
                                    categorical_indicator[X.columns.get_loc(col)] = True

                            categorical_columns = X.columns[
                                list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
                            
                            cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))  # 获得连续特征

                            cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
                            con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
                            
                            # 为所有样本生成一个标注
                            X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))

                            datafull = pd.read_csv(
                                Path("/home/lzy/data/MISS2/dataset/" + C.data.dataset + "/" + "mcar_" + C.data.dataset + "_" + str(
                                    0.0) + ".csv"))
                            datamiss = pd.read_csv(out)
                            
                            datamiss, indicator = sampling(datafull, datamiss, C.data.ipsnum, method='feature')
                            ips = compute_ips(datamiss[:, :-1], indicator[:, :-1], num=C.data.ipsnum, method='xgb')
                            ips = torch.tensor(ips, device=device)  # 添加device参数
                            softm = nn.Softmax(dim=1)
                            ips = softm(ips)
                            
                            cat_ips = ips[:, cat_idxs]
                            num_ips = ips[:, con_idxs]
                            cat_nan_mask = nan_mask[:, cat_idxs]
                            num_nan_mask = nan_mask[:, con_idxs]
                            train_index = X[X.Set == "train"].index
                            valid_index = X[X.Set == "valid"].index
                            test_index = X[X.Set == "test"].index

                            X = X.drop(columns=['Set']).to_numpy()
                            num_ips_dict = {'train': num_ips[train_index, :], 'val': num_ips[valid_index, :],
                                            'test': num_ips[test_index, :]}
                            cat_ips_dict = {'train': cat_ips[train_index, :], 'val': cat_ips[valid_index, :],
                                            'test': cat_ips[test_index, :]}
                            num_nan_mask_dict = {'train': num_nan_mask[train_index, :], 'val': num_nan_mask[valid_index, :],
                                                 'test': num_nan_mask[test_index, :]}
                            cat_nan_mask_dict = {'train': cat_nan_mask[train_index, :], 'val': cat_nan_mask[valid_index, :],
                                                 'test': cat_nan_mask[test_index, :]}
                            # %%
                            # %%
                            # 构建数据集对象 D，包含特征、标签等
                            D = lib.build_dataset(C.data.path, C.data.T, C.data.T_cache, cat_nan_mask_dict)
                            
                            """
                            lzy begin
                            两套数据划分的方式有冲突
                            对划分方式进行修正
                            """
                            # 检查并修正索引一致性问题
                            for part in ['train', 'test', 'val']:
                                expected_size = D.size(part)
                                actual_size = len(num_nan_mask_dict[part])
                                
                                if expected_size != actual_size:
                                    print(f"WARNING: Size mismatch for {part}: expected {expected_size}, got {actual_size}")
                                    # 截断或填充到正确大小
                                    if expected_size < actual_size:
                                        # 截断 - 保持tensor类型
                                        num_nan_mask_dict[part] = num_nan_mask_dict[part][:expected_size]
                                        cat_nan_mask_dict[part] = cat_nan_mask_dict[part][:expected_size]
                                        num_ips_dict[part] = num_ips_dict[part][:expected_size]
                                        cat_ips_dict[part] = cat_ips_dict[part][:expected_size]
                                    else:
                                        # 填充（使用最后一个样本重复） - 使用torch操作保持tensor类型
                                        deficit = expected_size - actual_size
                                        
                                        # 获取最后一个样本并重复
                                        last_num_mask = num_nan_mask_dict[part][-1:].repeat(deficit, 1)
                                        last_cat_mask = cat_nan_mask_dict[part][-1:].repeat(deficit, 1)
                                        last_num_ips = num_ips_dict[part][-1:].repeat(deficit, 1)
                                        last_cat_ips = cat_ips_dict[part][-1:].repeat(deficit, 1)
                                        
                                        # 使用torch.cat拼接
                                        num_nan_mask_dict[part] = torch.cat([num_nan_mask_dict[part], last_num_mask], dim=0)
                                        cat_nan_mask_dict[part] = torch.cat([cat_nan_mask_dict[part], last_cat_mask], dim=0)
                                        num_ips_dict[part] = torch.cat([num_ips_dict[part], last_num_ips], dim=0)
                                        cat_ips_dict[part] = torch.cat([cat_ips_dict[part], last_cat_ips], dim=0)
                            """
                            lzy end
                            """

                    
                            report['prediction_type'] = None if D.is_regression else 'logits'
                            lib.dump_pickle(D.y_info, output / 'y_info.pickle')


                            """
                            分箱 begins
                            直接使用数值求embedding 不同数之间的embedding向量差异可能很大
                            分箱后降低对此类情况的敏感度，鲁棒性更好
                            """
                            if C.bins is None:
                                bin_edges = None
                                bins = None
                                bin_values = None
                                n_bins = None
                            else:
                                # 对数值型特征进行分箱
                                print('\nRunning bin-based encoding...')
                                assert D.X_num is not None
                                bin_edges = []

                                # 初始化存储分箱结果的字典，分别为不同数据集部分（如'train', 'val', 'test'）
                                _bins = {x: [] for x in D.X_num}
                                _bin_values = {x: [] for x in D.X_num}

                                rng = np.random.default_rng(seed)

                                # 遍历每一个数值型特征
                                for feature_idx in trange(D.n_num_features):
                                    train_column = D.X_num['train'][:, feature_idx].copy()

                                    # 如果设置了子采样（subsample），则只用部分数据分箱 实际未使用
                                    if C.bins.subsample is not None:
                                        subsample_n = (
                                            C.bins.subsample
                                            if isinstance(C.bins.subsample, int)
                                            else int(C.bins.subsample * D.size('train'))
                                        )
                                        subsample_idx = rng.choice(len(train_column), subsample_n, replace=False)
                                        train_column = train_column[subsample_idx]
                                    else:
                                        subsample_idx = None

                                    if C.bins.tree is not None:
                                        # 注意，这里_y实际上是label，但后续从没有用上过
                                        _y = D.y['train']
                                        if subsample_idx is not None:
                                            _y = _y[subsample_idx]
                                        tree = (
                                            (DecisionTreeRegressor if D.is_regression else DecisionTreeClassifier)(
                                                max_leaf_nodes=C.bins.count, **C.bins.tree
                                            )

                                            # 实际应该是.fit(train_column.reshape(-1, 1), _y)
                                            # .fit(train_column.reshape(-1, 1), D.y['train'])
                                            .fit(train_column.reshape(-1, 1), _y)
                                            .tree_  # 获取树的底层实现
                                        )
                                        del _y
                                        tree_thresholds = []
                                        for node_id in range(tree.node_count):
                                            # the following condition is True only for split nodes
                                            # See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
                                            # 只收集分裂节点（对分类有贡献）的阈值
                                            if tree.children_left[node_id] != tree.children_right[node_id]:
                                                tree_thresholds.append(tree.threshold[node_id])
                                        tree_thresholds.append(train_column.min())
                                        tree_thresholds.append(train_column.max())
                                        bin_edges.append(np.array(sorted(set(tree_thresholds))))  # 对于每个feature都有，故最终bin-edges中有n_features个元素
                                    else:
                                        # 直接使用分位数进行分箱
                                        feature_n_bins = min(C.bins.count, len(np.unique(train_column)))
                                        quantiles = np.linspace(
                                            0.0, 1.0, feature_n_bins + 1
                                        )  # includes 0.0 and 1.0
                                        bin_edges.append(np.unique(np.quantile(train_column, quantiles)))


                                    # 生成分箱编号
                                    for part in D.X_num:  # 遍历数据集的每个部分（如'train', 'val', 'test'）
                                        _bins[part].append(
                                            np.digitize(  # 将连续值分配到对应的分箱区间
                                                D.X_num[part][:, feature_idx],  # 需要分箱的数值，按照分箱的范围返回一个代表分箱编号的数组
                                                np.r_[-np.inf, bin_edges[feature_idx][1:-1], np.inf],  # 按行拼接 参数含义为分箱的范围
                                            ).astype(np.int32)
                                            - 1  # 使得分箱编号从0开始
                                        )

                                        # 不同的分箱编码方式 
                                        if C.bins.encoding == 'binary':  # 对每个样本，分箱后的特征值都设为 1。这种编码方式通常用于 one-hot 编码等场景，表示“该样本属于某个分箱”即可，不关心具体数值。
                                            _bin_values[part].append(np.ones_like(D.X_num[part][:, feature_idx]))  # 所有_bin_values值都是1,即代表所有样本都在分箱中
                                        elif C.bins.encoding == 'piecewise-linear':  # 代表样本在具体哪个分箱中
                                            feature_bin_sizes = (
                                                    bin_edges[feature_idx][1:] - bin_edges[feature_idx][:-1]  # 这个feature对应的相邻两个分箱边界之间的距离，即每个分箱区间的宽度
                                            )
                                            part_feature_bins = _bins[part][feature_idx]  # _bins 分箱编号
                                            _bin_values[part].append(  # 约等于归一化
                                                (
                                                        D.X_num[part][:, feature_idx]  # 真实值
                                                        - bin_edges[feature_idx][part_feature_bins]  # 分箱边界值
                                                )
                                                / feature_bin_sizes[part_feature_bins]  # 分箱长度
                                            )
                                        else:
                                            assert C.bins.encoding == 'one-blob'

                                n_bins = max(map(len, bin_edges)) - 1

                                bins = {
                                    k: torch.as_tensor(np.stack(v, axis=1), dtype=torch.int64, device=device)
                                    for k, v in _bins.items()
                                }
                                del _bins

                                bin_values = (
                                    {
                                        k: torch.as_tensor(np.stack(v, axis=1), dtype=torch.float32, device=device)
                                        for k, v in _bin_values.items()
                                    }
                                    if _bin_values['train']
                                    else None
                                )
                                del _bin_values
                                lib.dump_pickle(bin_edges, output / 'bin_edges.pickle')
                                bin_edges = [torch.tensor(x, dtype=torch.float32, device=device) for x in bin_edges]
                                print()
                            """
                            分箱 ends
                            得到 bins 和 bin_values
                            """

                            X_num, X_cat, Y = lib.prepare_tensors(D, device=device)
                            if C.bins is not None and C.bins.encoding != 'one-blob':
                                pretrain_X_num = X_num
                                X_num = None

                            zero.hardware.free_memory()

                            loss_fn = lib.get_loss_fn(D.task_type)

                            model = (FlatModel if C.is_mlp or C.is_resnet else NonFlatModel)(C, D, n_bins).to(
                                device
                            )
                            # if torch.cuda.device_count() > 1:
                            #     print('Using nn.DataParallel')
                            #     model = nn.DataParallel(model)  # type: ignore[code]
                            report['n_parameters'] = lib.get_n_parameters(model)

                            optimizer = lib.make_optimizer(
                                asdict(C.training), lib.split_parameters_by_weight_decay(model)
                            )

                            stream = zero.Stream(
                                zero.data.IndexLoader(D.size('train'), C.training.batch_size, True, device=device)
                            )

                            report['epoch_size'] = math.ceil(D.size('train') / C.training.batch_size)
 
                            progress = zero.ProgressTracker(C.training.patience)  # 早停监控
                            progress_pretrain = zero.ProgressTracker(C.training.patience)
                            training_log = {}
                            checkpoint_path = output / 'checkpoint.pt'
                            eval_batch_size = C.training.eval_batch_size
                            chunk_size = None

                            # 记录训练中的最小值
                            min_pretrain_loss = float("inf")
                            min_pretrain_loss_idx = 0
                            
                            test_acc_list = []
                            timer = lib.Timer.launch()

                            for epoch in stream.epochs(20):
                                loss = 0
                                running_loss = 0
                                epoch_losses = []

                                pretrain_optimizer = optim.AdamW(lib.split_parameters_by_weight_decay(model), lr=0.0001)

                                for batch_idx in epoch:
                                    optimizer.zero_grad()

                                    if X_cat is not None:
                                        batch_X_cat = X_cat['train'][batch_idx]
                                        cat_mask = cat_nan_mask_dict['train'][batch_idx]
                                        cat_ips = cat_ips_dict['train'][batch_idx]

                                    batch_X_num = pretrain_X_num['train'][batch_idx]
                                    batch_y = Y['train'][batch_idx]
                                    num_mask = num_nan_mask_dict['train'][batch_idx]

                                    num_ips = num_ips_dict['train'][batch_idx]


                                    # 合并两种特征
                                    if X_cat is not None:
                                        ips = torch.concat([num_ips, cat_ips], 1)
                                        mask = torch.concat([num_mask, cat_mask], 1)
                                    else:
                                        ips = num_ips
                                        mask = num_mask
                                        
                                    # 移除最后一列   
                                    ips = ips[:, :-1]
                                    mask = mask[:, :-1]
                                    
                                    # 确保ips和mask在正确的设备上
                                    ips = ips.to(device)
                                    mask = mask.to(device)

                                    x = model._encode_input((
                                        encode('train', batch_idx)
                                        if C.bins is not None
                                        else X_num['train'][batch_idx]
                                        if X_num is not None
                                        else None
                                    ),
                                        None if X_cat is None else X_cat['train'][batch_idx], )

                                    x = torch.concat([x_ for x_ in x if x_ is not None], 1)
                                    
                                    # 对比学习 使得模型学习稳定的特征表示 同一样本经过两次不同的前向传播也应该得到相似的特征表示
                                    aug_features_1 = pretrain_module(model, x, ips, mask)
                                    aug_features_2 = pretrain_module(model, x, ips, mask)
                                    aug_features_1 = (
                                                aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,
                                                                                                                    2)
                                    aug_features_2 = (
                                                aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,
                                                                                                                    2)
                                    aug_features_1 = model.pt_mlp(aug_features_1)
                                    aug_features_2 = model.pt_mlp(aug_features_2)
                                    # nce_temp = 0.7  # 原
                                    nce_temp = torch.tensor(0.7, device=aug_features_1.device, dtype=torch.float32)
                                    logits_per_aug1 = aug_features_1 @ aug_features_2.t() / nce_temp
                                    logits_per_aug2 = aug_features_2 @ aug_features_1.t() / nce_temp
                                    targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                                    # print(logits_per_aug1.shape)
                                    # print(targets)
                                    loss_1 = criterion1(logits_per_aug1, targets)  # 交叉熵
                                    loss_2 = criterion1(logits_per_aug2, targets)
                                    # loss = 1 * (loss_1 + loss_2) / 2  # 原

                                    # 对比学习损失
                                    loss = (loss_1 + loss_2) / torch.tensor(2.0, device=loss_1.device, dtype=torch.float32)

                                    outs = pretrain_module(model, x, ips, mask)
                                    con_outs = outs[:, :D.n_num_features]
                                    if X_cat is not None:
                                        cat_outs = outs[:, D.n_num_features:]
                                        cat_outs = model.mlp1(cat_outs)
                                    con_outs = model.mlp2(con_outs)
                                    if len(con_outs) > 0:
                                        con_outs = torch.cat(con_outs, dim=1)
                                        l2 = criterion2(con_outs[num_mask == 0], batch_X_num[num_mask == 0])  # MSE
                                    else:
                                        # l2 = 0  # 原
                                        l2 = torch.tensor(0.0, device=batch_X_num.device, dtype=torch.float32)
                                        
                                    l1 = torch.tensor(0.0, device=batch_X_num.device, dtype=torch.float32)                                      
                                    # l1 = 0  # 原

                                    # import ipdb; ipdb.set_trace()
                                    if X_cat is not None:
                                        n_cat = batch_X_cat.shape[-1]
                                        for j in range(1, n_cat):
                                            log_x = criterion3(cat_outs[j]) 
                                            log_x = log_x[range(cat_outs[j].shape[0]), batch_X_cat[:, j]]
                                            # log_x[cat_mask[:, j] == 1] = 0  # 原

                                            # 避免原地操作，使用torch.where替代
                                            mask_condition = (cat_mask[:, j] == 1).to(log_x.device)  # 将条件转换为与log_x相同的设备
                                            log_x = torch.where(mask_condition, torch.tensor(0.0, device=log_x.device, dtype=log_x.dtype), log_x) #lzy

                                            # l1 += abs(sum(log_x) / cat_outs[j].shape[0])  # 原
                                            l1 += torch.abs(torch.sum(log_x) / torch.tensor(cat_outs[j].shape[0], device=log_x.device, dtype=torch.float32))
                                    # loss += 1 * l1 + 1 * l2  # 原
                                    loss = loss + l1 + l2  # lzy
                                    loss.backward()
                                    optimizer.step()
                                    running_loss += loss.item()
                                print(
                                    f'Epoch: {stream.epoch}, Running Loss: {running_loss}')
                                progress_pretrain = zero.ProgressTracker(C.training.patience)
                                progress_pretrain.update(running_loss)
                                if progress_pretrain.success:
                                    print('New best epoch!')
                                    report['best_epoch'] = epoch
                                    save_checkpoint()

                                elif progress_pretrain.fail:
                                    break

                            model.load_state_dict(torch.load(checkpoint_path)['model'])
                            for epoch in stream.epochs(C.training.n_epochs):
                                print_epoch_info()

                                model.train()
                                epoch_losses = []
                                for batch_idx in epoch:
                                    loss, new_chunk_size = lib.train_with_auto_virtual_batch(
                                        optimizer,
                                        loss_fn,
                                        lambda x: (apply_model('train', x), Y['train'][x]),
                                        batch_idx,
                                        chunk_size or C.training.batch_size,
                                    )
                                    epoch_losses.append(loss.detach())
                                    if new_chunk_size and new_chunk_size < (chunk_size or C.training.batch_size):
                                        report['chunk_size'] = chunk_size = new_chunk_size
                                        print('New chunk size:', chunk_size)

                                epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
                                metrics, predictions = evaluate(['val', 'test'])
                                lib.update_training_log(
                                    training_log,
                                    {
                                        'train_loss': epoch_losses,
                                        'mean_train_loss': mean_loss,
                                        'time': timer(),
                                    },
                                    metrics,
                                )
                                # print("[valauc]", metrics['val']['roc_auc'], "testauc", metrics['test']['roc_auc'])
                                print(f'\n{lib.format_scores(metrics)} [loss] {mean_loss:.3f}')
                                test_acc_list.append([metrics['test']['score'], metrics['test']['score2']])
                                progress.update(metrics['val']['score'])
                                if progress.success:
                                    print('New best epoch!')
                                    report['best_epoch'] = stream.epoch
                                    report['metrics'] = metrics
                                    save_checkpoint()
                                    lib.dump_predictions(predictions, output)

                                elif progress.fail or not are_valid_predictions(predictions):
                                    break
                            max_value = max(test_acc_list, key=lambda x: x[0])
                            print("best_test", max_value)
                            model.load_state_dict(torch.load(checkpoint_path)['model'])
                            report['metrics'], predictions = evaluate(['train', 'val', 'test'])
                            lib.dump_predictions(predictions, output)
                            report['time'] = str(timer)
                            save_checkpoint()
                            re = lib.finish(output, report)
                            print("time_cost", time.time() - start_time)
                            result = {"data": data_name, "missingrate": missingrate, "type": missingtype,
                                      "time": time.time() - start_time,
                                      "seed": seed,
                                      "result": re,
                                      "best_test_auroc": max_value[1],
                                      "best_test_accuracy": max_value[0], }
                            # ,"pretrain_epoch": pretrain_epoch,"pretrain_loss_con": pretrain_loss_con,"pretrain_loss_deno": pretrain_loss_deno,"pretrain_loss": pretrain_loss}

                            all_result.append(result)
                            # out = Path("/data/lsw/result/tabular_allresult" + t_time + ".json")
                            out = Path("/home/lzy/data/MISS2/result/tabular_allresult" + t_time + ".json")
                            
                            # 自动创建目录（如果不存在）
                            out.parent.mkdir(parents=True, exist_ok=True)            
                            with open(out, "w") as f:
                                json.dump(all_result, f, indent=4)