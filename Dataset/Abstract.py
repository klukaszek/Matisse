"""Abstract base class for JAX/Grain datasets."""
from abc import ABC, abstractmethod
import grain.python as pygrain
from typing import Iterator, Any


class GrainDataset(pygrain.RandomAccessDataSource, ABC):
    """Abstract base class for Grain datasets.

    Inherits from Grain's RandomAccessDataSource to provide efficient
    random access data loading for JAX training loops.
    """

    def __init__(self, params: dict, retina):
        """Initialize dataset.

        Args:
            params: Configuration parameters
            retina: RetinaModel instance for getting required image resolution
        """
        super().__init__()
        self.params = params
        self.retina = retina

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get a single sample from the dataset.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Sample data
        """
        pass
