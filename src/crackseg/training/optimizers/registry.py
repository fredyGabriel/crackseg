import torch.optim

_OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {}


def register_optimizer(name: str):
    def decorator(cls):
        _OPTIMIZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_optimizer(name: str) -> type[torch.optim.Optimizer]:
    return _OPTIMIZER_REGISTRY[name]


def list_optimizers() -> list:
    return list(_OPTIMIZER_REGISTRY.keys())
