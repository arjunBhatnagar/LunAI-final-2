"""Load user-supplied models from ``module:attribute`` specifications."""
from __future__ import annotations

import importlib
import inspect
from typing import Any


def load_object(specification: str, setting_name: str) -> Any:
    """Import ``module:attribute``; classes are instantiated with no arguments."""
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"{setting_name} must use the 'module:attribute' format, got {specification!r}.")
    module = importlib.import_module(module_name)
    target = module
    for part in attribute.split("."):
        target = getattr(target, part, None)
        if target is None:
            raise ValueError(f"{setting_name}: {attribute!r} was not found in module {module_name!r}.")
    return target() if inspect.isclass(target) else target
