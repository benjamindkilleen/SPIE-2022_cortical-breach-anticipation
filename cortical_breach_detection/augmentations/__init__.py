import traceback
from typing import Union

import kornia.augmentation as K
from omegaconf import DictConfig
from omegaconf import ListConfig
from torch import nn


AUGMENTATION_REGISTRY = {}
AUGMENTATION_REGISTRY_TB = {}
AUGMENTATION_CLASS_NAMES = set()
AUGMENTATION_CLASS_NAMES_TB = {}


def build_augmentation(config: Union[DictConfig, ListConfig]) -> nn.Module:
    if config is None:
        return nn.Identity()
    elif isinstance(config, (list, ListConfig)):
        name = "Sequential"
        args = map(build_augmentation, config)
        kwargs = {}
    else:
        assert "name" in config, f"name not provided for augmentation: {config}"
        name = config["name"]
        args = map(build_augmentation, config["augmentations"]) if "augmentations" in config else []
        kwargs = {
            k: (tuple(v) if isinstance(v, (list, ListConfig)) else v)
            for k, v in config.items()
            if k != "name" and k != "augmentations"
        }

    if name in AUGMENTATION_REGISTRY:
        augmentation = AUGMENTATION_REGISTRY[name](**kwargs)
    else:
        # the name should be available in kornia.augmentations
        if not (hasattr(K, name) or hasattr(nn, name)):
            name = name.title().replace("_", "")

        if hasattr(K, name):
            augmentation = getattr(K, name)(*args, **kwargs)
        elif hasattr(nn, name):
            augmentation = getattr(nn, name)(*args, **kwargs)
        else:
            raise KeyError(
                f"{name} is not a Kornia augmentation or nn Module, nor is it registered."
            )

    return augmentation


def register_augmentation(name):
    """Registers an augmentation.
    This decorator allows vertview to instantiate an augmentation
    from a configuration file. To use it, apply this decorator to an
    AugmentationBase2D subclass, like this:
    .. code-block:: python
     @register_augmentation("my_augmentation")
     class MyAugmentation(AugmentationBase2D):
          ...
    """

    def register_augmentation_cls(cls):
        if name in AUGMENTATION_REGISTRY:
            msg = "Cannot register duplicate augmentation ({}). Already registered at \n{}\n"
            raise ValueError(msg.format(name, AUGMENTATION_REGISTRY_TB[name]))
        if not issubclass(cls, nn.Module):
            raise ValueError(
                "Augmentation ({}: {}) must extend torch.nn.Module".format(name, cls.__name__)
            )
        tb = "".join(traceback.format_stack())
        AUGMENTATION_REGISTRY[name] = cls
        AUGMENTATION_CLASS_NAMES.add(cls.__name__)
        AUGMENTATION_REGISTRY_TB[name] = tb
        AUGMENTATION_CLASS_NAMES_TB[cls.__name__] = tb
        return cls

    return register_augmentation_cls


__all__ = ["build_augmentation", "register_augmentation"]
