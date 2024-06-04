from dataclasses import dataclass, field, fields
from lightgbm import Dataset as LGBMDataSet
from lightgbm import early_stopping, cv, LGBMClassifier
import optuna
from optuna.samplers import TPESampler
import numpy as np
from typing import List, Any, Callable, get_type_hints


# TODO define "strategy" classes with strategies such as deep tree, longer tree, regularization ... which would use
#   the base lightGBM classes


@dataclass
class LightGBMParametersBase:
    is_unbalanced: bool = field(init=False)
    num_thread: int = field(init=False)
    tree_learner: str = field(init=False)
    feature_fraction: float = field(init=False)
    sigmoid: float = field(init=False)
    num_leaves: int = field(init=False)
    lambda_l2: float = field(init=False)
    min_sum_hessian: int = field(init=False)
    bagging_fraction: float = field(init=False)
    sigmoid: float = field(init=False)
    learning_rate: int = field(default=1e-1)
    task: str = field(default="train")
    boosting: str = field(default="gbdt")
    device_type: str = field(default="cpu")
    seed: int = field(default=1010)
    verbosity: int = field(default=0)
    first_metric_only: bool = field(default=True)
    data_sample_strategy: str = field(default="goss")
    boost_from_average: bool = field(default=True)
    extra_trees: bool = field(default=True)
    is_provide_training_metric: bool = field(default=True)

    def __post_init__(self):
        self.params = {f.name: getattr(self, f.name) for f in fields(self) if f.name in self.__dict__.keys()}
        self._full_params = [f.name for f in fields(self)]

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
        # TODO: Change trial.suggest_* by a generic function from an "OptimizerBackendClass"
        #   so one can use other optimizer than optuna
        if trial:
            kwargs = self._verify_kwargs_to_field(**kwargs)
            for k, v in kwargs.items():
                hyper_param_type = get_type_hints(self).get(k)
                if hyper_param_type is int:
                    transform_fct = trial.suggest_int
                if hyper_param_type is float:
                    transform_fct = trial.suggest_float
                else:
                    transform_fct = trial.suggest_categorial
                    pass
                if isinstance(v, list):
                    self.params |= {k: transform_fct(k, *v)}
                else:
                    self.params |= {k: transform_fct(k, **v)}
        elif refit:
            self.params |= {"task": "refit"}
        else:
            self.params |= {"task": "predict"}
        return self.params

    def _suggest(self, name, fct, *args, **kwargs):
        self.params |= fct(name, *args, **kwargs)

    def _verify_kwargs_to_field(self, **kwargs) -> dict:
        return {k: v for k, v in kwargs.items() if k in self._full_params}


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
