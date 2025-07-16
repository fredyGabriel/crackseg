from typing import Any

from torch import nn

from crackseg.training.losses.loss_registry_setup import loss_registry

from .base_loss import SegmentationLoss


class WeightedSumLoss(SegmentationLoss):
    """
    Weighted sum of multiple losses (can be recursive).
    """

    def __init__(
        self,
        components: list[nn.Module],
        weights: list[float] | None = None,
    ):
        super().__init__()
        self.components = nn.ModuleList(components)
        if weights is None:
            self.weights = [1.0 / len(components)] * len(components)
        else:
            if len(weights) != len(components):
                raise ValueError(
                    "The number of weights must match the number of "
                    "components."
                )
            s = sum(weights)
            if s <= 0:
                raise ValueError("The sum of the weights must be positive.")
            self.weights = [w / s for w in weights]

    def forward(self, pred: Any, target: Any) -> Any:
        return sum(
            w * c(pred, target)
            for w, c in zip(self.weights, self.components, strict=False)
        )


class ProductLoss(SegmentationLoss):
    """
    Product of multiple losses (can be recursive).
    """

    def __init__(self, components: list[nn.Module]):
        super().__init__()
        self.components = nn.ModuleList(components)

    def forward(self, pred: Any, target: Any) -> Any:
        result = self.components[0](pred, target)
        for c in self.components[1:]:
            result = result * c(pred, target)
        return result


def parse_loss_config(config: dict[str, Any]) -> nn.Module:
    """
    Builds a hierarchy of losses recursively from a nested configuration.
    """
    if "type" in config:
        comb_type = config["type"]
        components_cfg = config["components"]
        weights = config.get("weights", None)
        child_losses = [parse_loss_config(child) for child in components_cfg]
        if comb_type == "sum":
            return WeightedSumLoss(child_losses, weights)
        elif comb_type == "product":
            return ProductLoss(child_losses)
        else:
            raise ValueError(f"Tipo de combinación no soportado: {comb_type}")
    elif "name" in config:
        loss_name = config["name"]
        params = config.get("params", {})
        return loss_registry.instantiate(loss_name, **params)
    else:
        raise ValueError(
            "Nodo de configuración inválido: falta 'type' o 'name'."
        )
