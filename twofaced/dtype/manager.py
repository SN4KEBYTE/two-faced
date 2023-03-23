from types import MappingProxyType
from typing import Any

from twofaced.backend import TORCH, NUMPY, get_backend
from twofaced.dtype.spec import NumpyDtypeSpec, TorchDtypeSpec


class DtypeManager:
    def __init__(
        self,
    ) -> None:
        self._specs = {
            NUMPY: NumpyDtypeSpec(),
            TORCH: TorchDtypeSpec(),
        }

    # floating point data types
    @property
    def float16(
        self,
    ) -> Any:
        return self._specs[get_backend()].float16

    @property
    def float32(
        self,
    ) -> Any:
        return self._specs[get_backend()].float32

    @property
    def float64(
        self,
    ) -> Any:
        return self._specs[get_backend()].float64

    # integer data types
    @property
    def bool(
        self,
    ) -> Any:
        return self._specs[get_backend()].bool

    # complex data types
    @property
    def complex32(
        self,
    ) -> Any:
        return self._specs[get_backend()].complex32

    @property
    def complex64(
        self,
    ) -> Any:
        return self._specs[get_backend()].complex64
