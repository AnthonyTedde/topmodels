"""
Module containing various decorators.

Available Decorators:
    * `require_not_none`: Ensures specified attributes of a classe instance
                            are not set to `None` before executing 
                            the method.
"""

from functools import wraps

def require_not_none(*attr_names):
    """
    Decorator that checks specified attributes of an instance are not None before executing the decorated method.

    Args:
        attr_names (str): One or more attribute names to verify on the instance (self).

    Raises:
        ValueError: If any of the specified attributes are None.

    Returns:
        _type_: _description_
    """    
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            missing_attrs = [
                attr for attr in attr_names if getattr(self, attr, None) is None
            ]
            if missing_attrs:
                raise ValueError(
                    f"The following attributes are mandatory: {', '.join(missing_attrs)}"
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
