from __future__ import annotations

import os
from setuptools import Extension, setup


def is_windows() -> bool:
    return os.name == "nt"


extra_compile_args: list[str] = []
extra_link_args: list[str] = []

if not is_windows():
    extra_compile_args.append("-std=c99")
    # exp/log are used in stochastic decoding
    extra_link_args.append("-lm")

ext_modules = [
    Extension(
        "mmjp._mmjp",
        sources=[
            "mmjp/_mmjp.c",
            "tools/mmjp_model.c",
            "npycrf_lite/npycrf_lite.c",
            "double_array/double_array_trie.c",
        ],
        include_dirs=[".", "tools", "npycrf_lite", "double_array"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(ext_modules=ext_modules)
