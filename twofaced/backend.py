from dataclasses import dataclass
from types import MappingProxyType, ModuleType

import numpy as np
import torch


NUMPY = 'numpy'
TORCH = 'torch'
_ALLOWED_BACKENDS = MappingProxyType({
    NUMPY: np,
    TORCH: torch,
})


@dataclass
class Config:
    backend: ModuleType = np
    backend_name: str = NUMPY

    def set_backend(
        self,
        backend: str,
    ) -> None:
        if backend not in _ALLOWED_BACKENDS:
            raise ValueError(f'unknown backend "{backend}". Possible backends are {", ".join(_ALLOWED_BACKENDS)}')

        self.backend_name = backend
        self.backend = _ALLOWED_BACKENDS[backend]

    @property
    def current_backend(
        self,
    ) -> str:
        return self.backend_name


_config = Config()


def set_backend(
    backend: str,
) -> None:
    _config.set_backend(backend)


def get_backend() -> str:
    return _config.current_backend
