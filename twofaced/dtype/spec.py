from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class DtypeSpec(ABC):
    # floating point data types
    @property
    @abstractmethod
    def float16(
        self,
    ) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def float32(
        self,
    ) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def float64(
        self,
    ) -> Any:
        raise NotImplementedError

    # integer data types
    @property
    @abstractmethod
    def bool(
        self,
    ) -> Any:
        raise NotImplementedError

    # complex data types
    @property
    @abstractmethod
    def complex32(
        self,
    ) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def complex64(
        self,
    ) -> Any:
        raise NotImplementedError


class TorchDtypeSpec(DtypeSpec):
    # floating point data types
    @property
    def float16(
        self,
    ) -> Any:
        return torch.float16

    @property
    def float32(
        self,
    ) -> Any:
        return torch.float32

    @property
    def float64(
        self,
    ) -> Any:
        return torch.float64

    # integer data types
    @property
    def bool(
        self,
    ) -> Any:
        return torch.bool

    # complex data types
    @property
    def complex32(
        self,
    ) -> Any:
        return torch.complex32

    @property
    def complex64(
        self,
    ) -> Any:
        return torch.complex64


class NumpyDtypeSpec(DtypeSpec):
    # floating point data types
    @property
    def float16(
        self,
    ) -> Any:
        return np.float16

    @property
    def float32(
        self,
    ) -> Any:
        return np.float32

    @property
    def float64(
        self,
    ) -> Any:
        return np.float64

    # integer data types
    @property
    def bool(
        self,
    ) -> Any:
        return np.bool_

    # complex data types
    @property
    def complex32(
        self,
    ) -> Any:
        return np.csingle

    @property
    def complex64(
        self,
    ) -> Any:
        return np.cdouble
