"""Pretrained vision models and intermediate-feature extraction utilities.

This module provides a common loader for ResNet-18, ViT-B/32, DINOv2, and
CLIP, together with a wrapper that records outputs from compound blocks during
a forward pass. Models are returned in evaluation mode and moved to the
requested device.
"""

import os
import warnings
from typing import Any, Callable, Dict, Tuple

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="xFormers is disabled")
warnings.filterwarnings("ignore", message="xFormers is not available")

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision import transforms
from torchinfo import summary
import clip


def getModel(
    model_name: str,
    device: str | torch.device = "cpu",
    nofc: bool = True,
) -> Tuple[nn.Module, Callable[[Any], torch.Tensor]]:
    """Load a pretrained vision model and its input preprocessing pipeline.

    Args:
        model_name: One of ``resnet18``, ``vit_b_32``, ``dinov2_vits14``, or
            ``clip_vit_b_32``.
        device: Torch device name or device object on which to place the model.
        nofc: Replace the classifier head with ``Identity`` for ResNet-18 and
            ViT-B/32 when true, returning embeddings instead of class logits.

    Returns:
        A tuple containing the evaluation-mode model and its preprocessing
        callable.

    Raises:
        ValueError: If ``model_name`` is not supported.
    """
    # os.environ["XFORMERS_DISABLED"] = "0"

    device = torch.device(device)
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        preprocess = weights.transforms()
        model = resnet18(weights=weights)
        if nofc:
            model.fc = torch.nn.Identity()  # drop the last layer to get features instead of classification
    elif model_name == "vit_b_32":
        weights = ViT_B_32_Weights.DEFAULT
        preprocess = weights.transforms()
        model = vit_b_32(weights=weights)
        if nofc:
            model.heads = torch.nn.Identity()  # drop the last layer to get features instead of classification
    elif model_name == "dinov2_vits14":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == "clip_vit_b_32":
        model, preprocess = clip.load("ViT-B/32")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    model.eval() # avoid training mode, we just want to extract features
    model.to(device)
    return model, preprocess


class IntermediateFeatureExtractor(torch.nn.Module):
    """Run a model and collect the output at the end of each block.

    Only compound modules directly contained in a ``Sequential`` or
    ``ModuleList`` are hooked.  This captures block outputs (for example,
    ResNet residual blocks and Transformer encoder blocks) without collecting
    the operations inside those blocks.
    """
    def __init__(self, model: nn.Module) -> None:
        """Wrap ``model`` and register hooks on its compound child blocks.

        Args:
            model: Torch module whose block outputs should be collected during
                the forward pass.
        """
        super().__init__()
        self.model = model
        self.layer_features = {}
        self._hooks = []

        modules = dict(self.model.named_modules())
        for name, layer in modules.items():
            if not name or not list(layer.children()):
                continue

            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = modules.get(parent_name)
            if isinstance(parent, (torch.nn.Sequential, torch.nn.ModuleList)):
                self._hooks.append(
                    layer.register_forward_hook(self._save_feature(name))
                )

    @staticmethod
    def _detach(output: Any) -> Any:
        """Detach tensors recursively while preserving container structure."""
        if torch.is_tensor(output):
            return output.detach()
        if isinstance(output, tuple):
            return tuple(IntermediateFeatureExtractor._detach(item) for item in output)
        if isinstance(output, list):
            return [IntermediateFeatureExtractor._detach(item) for item in output]
        if isinstance(output, dict):
            return {key: IntermediateFeatureExtractor._detach(value)
                    for key, value in output.items()}
        return output

    def _save_feature(self, name: str) -> Callable[..., None]:
        """Create a forward hook that stores a detached module output."""
        def hook(*args: Any) -> None:
            output = args[-1]
            self.layer_features[name] = self._detach(output)

        return hook

    def forward(self, x: torch.Tensor) -> Tuple[Any, Dict[str, Any]]:
        """Run the model and return its output plus collected block features."""
        self.layer_features.clear()
        output = self.model(x)
        return output, self.layer_features

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks from the wrapped model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
