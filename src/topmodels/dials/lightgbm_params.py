from dataclasses import dataclass, field, fields


import fontTools.cu2qu.cu2qu

from lightgbm import Dataset as LGBMDataSet
from lightgbm import early_stopping, cv, LGBMClassifier
import optuna
from optuna.samplers import TPESampler
import numpy as np
from typing import List, Any, Callable, get_type_hints

from parameters import (
    BinaryParameter,
    CategoricalParameter,
    FloatingPointParameter,
    DiscreteParameter
)


# TODO define "strategy" classes with strategies such as deep tree, longer tree, regularization ... which would use
#   the base lightGBM classes


@dataclass
class LightGBMParametersBase:
    is_unbalanced: BinaryParameter = field(init=False)
    num_thread: DiscreteParameter = field(init=False)
    tree_learner: CategoricalParameter = field(init=False)
    feature_fraction: FloatingPointParameter = field(init=False)
    sigmoid: FloatingPointParameter = field(init=False)
    num_leaves: DiscreteParameter = field(init=False)
    lambda_l2: FloatingPointParameter = field(init=False)
    min_sum_hessian: DiscreteParameter = field(init=False)
    bagging_fraction: FloatingPointParameter = field(init=False)
    sigmoid: FloatingPointParameter = field(init=False)
    learning_rate: FloatingPointParameter = field(default=FloatingPointParameter(1e-1))
    task: CategoricalParameter = field(default=CategoricalParameter("train", choices=["train", "predict", "refit"]))
    boosting: CategoricalParameter = field(default=CategoricalParameter("gbdt", choices=["gbdt", "rf", "dart"]))
    device_type: CategoricalParameter = field(default=CategoricalParameter("cpu", choices=["cpu", "cuda"]))
    seed: DiscreteParameter = field(default=DiscreteParameter(1010))
    verbosity: DiscreteParameter = field(default=DiscreteParameter(0))
    first_metric_only: BinaryParameter = field(default=BinaryParameter(True))
    data_sample_strategy: CategoricalParameter = field(default=CategoricalParameter("goss", choices=["bagging", "goss"]))
    boost_from_average: BinaryParameter = field(default=BinaryParameter(True))
    extra_trees: BinaryParameter = field(default=BinaryParameter(True))
    is_provide_training_metric: BinaryParameter = field(default=BinaryParameter(True))

    def __post_init__(self):
        self.params = {f.name: getattr(self, f.name) for f in fields(self) if f.name in self.__dict__.keys()}
        self._parameters_names = [f.name for f in fields(self)]

    def get_parameters_names(self):
        return [f.name for f in fields(self)]

    # TODO: Move `update_params` up to unspecialized method
    def update_params(self, **kwargs):
        # TODO: Consider update from after-training phase
        kwargs = self._verify_kwargs_to_field(**kwargs)
        # TODO: Validate the type of the kwargs,
        #  e.g., float, int, categorical expected
        #  should be float, int, or categorical

        for k, v in kwargs.items():
            setattr(self, k, v)

    # TODO: Push get_params up in the class hierarchy
    def get_params(self, trial=None, refit=False, **kwargs):

        if trial:
            kwargs = self._verify_kwargs_to_field(**kwargs)
            for k, v in kwargs.items():
                # TODO assign attribute using **kwargs
                setattr(self, k, v.suggest_value(trial=trial, name=k, *v))
        # TODO: Implement the return value, but what could the function return ?
        return self.params

    def _suggest(self, name, fct, *args, **kwargs):
        self.params |= fct(name, *args, **kwargs)

    def _verify_kwargs_to_field(self, **kwargs) -> dict:
        parameters_names = self.get_parameters_names()
        return {k: v for k, v in kwargs.items() if k in parameters_names}


@dataclass
class LightGBMParametersBinaryClassifier(LightGBMParametersBase):
    objective: str = field(default="binary")
    metric: List[str] = field(default_factory=lambda: ["binary_logloss", "binary_error", "auc", ])


@dataclass
class LightGBMParametersMulticlassClassifier(LightGBMParametersBase):
    objective: str = field(default="multiclass")
    num_class: int = field(init=False)
    metric: List[str] = field(default_factory=lambda: ["multi_logloss", "multi_error", "auc_mu", ])


@dataclass
class LightGBMParametersRegressor(LightGBMParametersBase):
    objective: str = field(default="regression")
    metric: List[str] = field(default_factory=lambda: ["rmse", "l2", "l1", ])
