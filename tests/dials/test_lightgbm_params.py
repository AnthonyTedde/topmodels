import pytest
from dataclasses import fields
from unittest.mock import Mock

from src.topmodels.dials.lightgbm_params import (
    LightGBMParametersBase,
    LightGBMParametersBinaryClassifier,
    LightGBMParametersMulticlassClassifier,
    LightGBMParametersRegressor,
)


class TestLightGBMParametersBase:

    def test_default_initialization(self):
        params = LightGBMParametersBase()
        assert params.learning_rate == 1e-1
        assert params.task == "train"
        assert params.objective == "multiclass"
        assert params.num_class == 2
        assert params.boosting == "gbdt"
        assert params.device_type == "cpu"
        assert params.seed == 1010
        assert params.verbosity == 0
        assert params.metric == ["multi_logloss", "multi_error", "auc_mu"]
        assert params.first_metric_only
        assert params.data_sample_strategy == "goss"
        assert params.boost_from_average
        assert params.extra_trees
        assert params.is_provide_training_metric
        assert params.params["learning_rate"] == 1e-1

    def test_update_params(self):
        params = LightGBMParametersBase()
        params.update_params(learning_rate=0.05, num_class=3, totallybullshitpara="toto")
        assert params.learning_rate == 0.05
        assert params.num_class == 3

    def test_update_params_and_accessed_by_name(self):
        params = LightGBMParametersBase()
        params.update_params(learning_rate=0.05, num_class=3, totallybullshitpara="toto")
        assert params.params["learning_rate"] == 0.05
        assert params.params["num_class"] == 3

    def test_get_params_without_trial(self):
        params = LightGBMParametersBase()
        result = params.get_params()
        assert result["task"] == "predict"

    def test_get_params_with_refit(self):
        params = LightGBMParametersBase()
        result = params.get_params(refit=True)
        assert result["task"] == "refit"

    def test_get_params_with_trial_args(self):
        trial = Mock()
        trial.suggest_float.side_effect = [0.5, 0.1, 0.8]
        trial.suggest_int.side_effect = [128, 5]

        params = LightGBMParametersBase()
        result = params.get_params(trial=trial,
                                   feature_fraction=[0.4, 1.0], sigmoid=[10e-4, 5.0],
                                   num_leaves=[8, 256])

        assert "feature_fraction" in result
        assert "sigmoid" in result
        assert "num_leaves" in result

    def test_get_params_with_trial_kwargs(self):
        trial = Mock()
        trial.suggest_float.side_effect = [0.5, 0.1, 0.8]
        trial.suggest_int.side_effect = [128, 5]

        params = LightGBMParametersBase()
        result = params.get_params(trial=trial,
                                   feature_fraction={"low": 0.4, "high": 1.0},
                                   sigmoid={"low": 10e-4, "high": 5.0},
                                   num_leaves={"low": 8, "high": 256})

        assert "feature_fraction" in result
        assert "sigmoid" in result
        assert "num_leaves" in result

    def test__verify_kwargs_to_field(self):
        params = LightGBMParametersBase()
        result = params._verify_kwargs_to_field(feature_fraction=[0.4, 1.0],
                                                foo=[1, 2], bar=[2, 3])
        assert "feature_fraction" in result.keys()
        assert "foo" not in result.keys()
        assert "bar" not in result.keys()


class TestLightGBMParametersBinaryClassifier:
    def test_default_initialization(self):
        # FAILURE: TODO: Implement _get_attr_
        binary_params = LightGBMParametersBinaryClassifier()
        assert binary_params.params["objective"] == "binary"
        assert sorted(binary_params.params["metric"]) == sorted(["binary_logloss", "binary_error", "auc", ])



class TestLightGBMParametersMulticlassClassifier:
    def test_default_initialization(self):
        multi_params = LightGBMParametersMulticlassClassifier()
        assert multi_params.params["objective"] == "multiclass"
        assert sorted(multi_params.params["metric"]) == sorted(["multi_logloss", "multi_error", "auc_mu", ])


class TestLightGBMParametersRegressor:
    def test_default_initialization(self):
        regressor_params = LightGBMParametersRegressor()
        assert regressor_params.params["objective"] == "regression"
        assert sorted(regressor_params.params["metric"]) == sorted(["rmse", "l2", "l1", ])

