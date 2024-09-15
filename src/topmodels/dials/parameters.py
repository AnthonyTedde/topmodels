from dataclasses import dataclass, Field
from abc import ABC, abstractmethod
from typing import Any, List
from optuna.trial import Trial


class ParameterBase(ABC):
    parameter: Any

    @abstractmethod
    def suggest_value(self, trial: Trial, name, *args, **kwargs):
        pass


@dataclass
class BinaryParameter(ParameterBase):
    parameter: bool

    def suggest_value(self, trial: Trial, name, *args, **kwargs):
        return trial.suggest_categorical(name=name, *args, **kwargs)


@dataclass
class DiscreteParameter(ParameterBase):
    parameter: int

    def suggest_value(self, trial: Trial, name, *args, **kwargs):
        return trial.suggest_int(name=name, *args, **kwargs)

@dataclass
class FloatingPointParameter(ParameterBase):
    parameter: float

    def suggest_value(self, trial: Trial, name, *args, **kwargs):
        return trial.suggest_float(name=name, *args, **kwargs)


@dataclass
class CategoricalParameter(ParameterBase):
    parameter: str
    choices: List[str] = Field(default=None)

    def suggest_value(self, trial: Trial, name, *args, **kwargs):
        return trial.suggest_categorical(name=name, *args, **kwargs)
