from abc import ABC
from typing import List, Tuple, Callable, Union
from numpy.typing import ArrayLike

from topmodels.utils.decorators import require_not_none


def make_property(metric_func, **func_kwargs):
    """
    Factory function to create a property that computes a metric using
    the attributes `self.y_true` and `self.y_pred` of the metric class.

    Args:
        metric_func (callable): The metric function to compute.
        **func_kwargs: Optional keyword arguments to pass to the metric function.

    Returns:
        property: A property that computes the metric when accessed.

    Author: AnthonyTedde
    """
    @require_not_none("y_true", "y_pred")
    def prop(self):
        return metric_func(y_true=self.y_true, y_pred=self.y_pred, **func_kwargs)

    return prop


def add_performance_metrics(metrics: List[Tuple[str, Callable, Union[dict, None]]]):
    """
    Class decorator that adds performance metric properties to a class.

    Args:
        metrics (List[Tuple[str, Callable, Union[dict, None]]]):
            A list of tuples, each containing:
                - prop_name (str): The name of the property to add to the class.
                - metric_func (Callable): A function that computes the metric.
                - func_kwargs (dict or None): Optional keyword arguments for the metric function.

    Returns:
        cls: The class with added metric properties.

    The decorator adds properties to the class for each specified metric.
    Each property computes the metric by calling `metric_func`
    with `self.y_true` and `self.y_pred`,
    passing any additional keyword arguments provided in `func_kwargs`.

    Example:
        from sklearn.metrics import accuracy_score, f1_score

        @add_performance_metrics([
            ('accuracy', accuracy_score, None),
            ('f1_macro', f1_score, {'average': 'macro'}),
            ('f1_micro', f1_score, {'average': 'micro'}),
        ])
        class ModelEvaluator:
            def __init__(self, y_true, y_pred):
                self.y_true = y_true
                self.y_pred = y_pred

        evaluator = ModelEvaluator(y_true=[0, 1], y_pred=[0, 1])
        print(evaluator.accuracy)   # Outputs the accuracy score
        print(evaluator.f1_macro)   # Outputs the macro-averaged F1 score
        print(evaluator.f1_micro)   # Outputs the micro-averaged F1 score

    Author: AnthonyTedde
    """
    def decorator(cls):
        cls._metrics = metrics.copy()
        for prop_name, metric_func, func_kwargs in metrics:
            setattr(cls, prop_name, make_property(metric_func, **func_kwargs or {}))
        return cls

    return decorator


class AbstractEvaluator(ABC):
    """
    Base class for evaluating models using customizable performance metrics.

    Author: AnthonyTedde

    """

    _metrics: List[Tuple[str, Callable, Union[dict, None]]]
    
    def __init__(self, y_true: ArrayLike | None, y_pred: ArrayLike | None):
        """
        Initializes the evaluator with true and predicted labels.

        Args:
            y_true (ArrayLike | None): The ground truth target values.
            y_pred (ArrayLike | None): The predicted target values.

        Author: AnthonyTedde

        """        
        self.y_true = y_true
        self.y_pred = y_pred
    
    @classmethod
    def add_metric(cls, 
                   prop_name: str, 
                   metric_func: Callable, 
                   **func_kwargs: Union[dict, None]):
        """
        Adds a new performance metric property to the class.

        This method allows you to dynamically add a new metric to the class.
        The metric will be accessible as a property with the name specified by `prop_name`.
        The property computes the metric using `metric_func` with `self.y_true` and `self.y_pred`.

        Args:
            prop_name (str): The name of the property to add to the class.
            metric_func (Callable): The function that computes the metric.

        Raises:
            ValueError: If the property name is already in use.

        Author: AnthonyTedde
        
        """                    
        if prop_name in (name for name, *_ in cls._metrics):
            raise ValueError(f'Property name {prop_name} already taken')
        cls._metrics.append((prop_name, metric_func, func_kwargs))
        setattr(cls, prop_name, make_property(metric_func, **func_kwargs or {}))
            
    @classmethod
    def remove_metric(cls, prop_name: str): 
        """
        Removes an existing performance metric property from the class.

        This method removes the metric property specified by `prop_name` from the class.
        It also removes the metric from the `_metrics` list.

        Args:
            prop_name (str): The name of the property to remove.

        Raises:
            ValueError: If the property name does not exist.

        Author: AnthonyTedde
        
        """        
        for idx, (name, *_) in enumerate(cls._metrics):
            if name == prop_name:
                del cls._metrics[idx]
                delattr(cls, prop_name)
                return
        raise ValueError(f'Property name {prop_name} does not exist')

    # @classmethod
    # def get_metrics(cls, ...):
    #     pass
    # ou __repr__ ... (?)