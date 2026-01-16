"""JAX/Grain-based data loading pipeline for Matisse."""
import importlib
import pkgutil
import os

# Registry for all dataset classes
_registry = {}


def register_class(name):
    """Decorator to register a dataset class."""
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator


def create_dataset(dataset_name: str, params: dict, retina):
    """Create an instance of the specified dataset.

    Args:
        dataset_name: Name of the dataset to create
        params: Configuration parameters
        retina: RetinaModel instance for getting required image resolution

    Returns:
        Grain DataLoader instance
    """
    if dataset_name not in _registry:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(_registry.keys())}"
        )
    return _registry[dataset_name](params, retina)


# Dynamically import all modules in this package
package_dir = os.path.dirname(__file__)
for module_info in pkgutil.iter_modules([package_dir]):
    module_name = module_info.name
    if not module_name.startswith('_'):
        importlib.import_module(f"{__name__}.{module_name}")
