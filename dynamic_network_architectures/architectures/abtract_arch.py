from copy import deepcopy
from typing import Literal, Sequence
from torch import nn
import torch
from dataclasses import dataclass

stem_mode = Literal["random_init", "exact_match", "repeat_to_match"]


class AbstractDynamicNetworkArchitectures(nn.Module):

    def __init__(self):
        super(AbstractDynamicNetworkArchitectures, self).__init__()
        # Key to the position holding all the encoder weights
        self.key_to_encoder: str
        # Key to the full stem -- Can be located within or outside the encoder
        self.key_to_stem: str
        # Not sure yet if we need anything but this -- but minor redundancy is okay I suppose
        # Key to the weights that are dependent on the input channels.
        #   Can hold multiple weights (e.g. for bad weight mappings like in this repo >.<' )
        self.keys_to_in_proj: Sequence[str]

    # def load_weights(
    #     self,
    #     to_be_loaded_state_dict: dict[str, torch.Tensor],
    #     to_be_loaded_key_to_encoder: str,
    #     to_be_loaded_key_to_stem: str,
    #     to_be_loaded_stem_adapt_keys: Sequence[str],
    #     load_encoder: bool,
    #     load_stem: stem_mode,
    # ):
    #     """
    #     Load weights into the network. This is a bit of a hack, but it works.
    #     :param to_be_loaded_state_dict: The state dict to load from
    #     :param load_encoder: Whether to load the encoder weights
    #     :param load_stem: Whether to load the stem weight -- and in which way.
    #     :return:
    #     """

    #     stem_part_of_encoder = self.key_to_stem.startswith(self.key_to_encoder)
    #     current_state_dict = deepcopy(self.state_dict())

    #     # -------------------------- Strip loaded state dict ------------------------- #
    #     encoder_state_dict = {}
    #     stem_state_dict = {}
    #     for k, v in to_be_loaded_state_dict.items():
    #         if k.startswith(to_be_loaded_key_to_encoder):
    #             new_k = deepcopy(k)
    #             new_k = new_k.replace(to_be_loaded_key_to_encoder, "")
    #             if new_k.startswith("."):
    #                 new_k = new_k[1:]
    #             encoder_state_dict[k.replace(to_be_loaded_key_to_encoder, "")] = v
    #         if k.startswith(to_be_loaded_key_to_stem):
    #             new_k = deepcopy(k)
    #             new_k = new_k.replace(to_be_loaded_key_to_stem, "")
    #             if new_k.startswith("."):
    #                 new_k = new_k[1:]
    #             stem_state_dict[new_k] = v

    #     # We keep the current state_dict to avoid
    #     if load_encoder:
    #         self.load_state_dict(to_be_loaded_state_dict, strict=False)
    #         return

    #     # Load the stem weights
    #     for key in self.key_to_stem_weights:
    #         if key in to_be_loaded_state_dict:
    #             self.state_dict()[key].copy_(to_be_loaded_state_dict[key])


def test_submodules_loadable(model: AbstractDynamicNetworkArchitectures):
    encoder_key = model.key_to_encoder
    stem_key = model.key_to_stem
    stem_weights_key = model.keys_to_in_proj
    # Check if the encoder submodule is loadable
    # Throws an error otherwise.
    _ = model.get_submodule(encoder_key)
    _ = model.get_submodule(stem_key)
    prev_shape = None
    for swk in stem_weights_key:
        stem_weights_submodule = model.get_submodule(swk).weight
        if prev_shape is None:
            prev_shape = stem_weights_submodule.shape
        else:
            assert stem_weights_submodule.shape == prev_shape, f"Stem weights submodule {swk} has different shape"
            prev_shape = stem_weights_submodule.shape
        assert stem_weights_submodule is not None, f"Stem weights submodule {swk} is not loadable"
    return
