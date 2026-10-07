from __future__ import annotations

import os

from setuptools import find_packages, setup


extension_modules = []
cmdclass = {}
if os.getenv("MXFLASHATTN_BUILD_NATIVE") == "1":
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    extension_modules.append(
        CppExtension(
            "mxflashattn._C",
            sources=[
                "csrc/mxmac/attention_forward.cpp",
                "csrc/mxmac/paged_kv.cpp",
                "csrc/mxmac/workspace.cpp",
            ],
            include_dirs=["csrc/mxmac"],
            extra_compile_args={"cxx": ["-O2", "-std=c++17"]},
        )
    )
    cmdclass["build_ext"] = BuildExtension


setup(
    packages=find_packages(include=["mxflashattn*", "benchmarks*", "integrations*"]),
    ext_modules=extension_modules,
    cmdclass=cmdclass,
)
