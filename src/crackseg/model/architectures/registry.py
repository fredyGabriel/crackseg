import torch.nn

_MODEL_REGISTRY: dict[str, type[torch.nn.Module]] = {}


def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str) -> type[torch.nn.Module]:
    return _MODEL_REGISTRY[name]


def list_models() -> list:
    return list(_MODEL_REGISTRY.keys())
