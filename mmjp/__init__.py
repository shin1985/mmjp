"""MMJP: Tiny Japanese tokenizer / subword segmenter.

This package provides Python bindings for the C99 core.

Example:
    >>> import mmjp
    >>> m = mmjp.Model("model.bin")
    >>> m.tokenize("東京都に住んでいます")
"""

from ._mmjp import Model  # noqa: F401

__all__ = ["Model"]

__version__ = "0.1.1"
