# Docstring Style Guide

PyADM1ODE follows the **Google Python Style Guide** for docstrings. This ensures consistent, readable, and automatically parseable documentation.

## General Format

```python
def function(arg1: int, arg2: str) -> bool:
    """
    Summary line.

    Extended description of the function and its behavior.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of the return value.

    Raises:
        ValueError: If arg1 is negative.
    """
```

## Classes

```python
class MyClass:
    """
    Summary line.

    Extended description.

    Attributes:
        attr1: Description of attr1.
    """
```

## Tools

We use `interrogate` to enforce docstring coverage and `mkdocstrings` to generate the API reference.
Current threshold is **95%**.
