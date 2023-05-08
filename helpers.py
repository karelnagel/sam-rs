import numpy as np
import json
import os
import torch
from segment_anything.build_sam import _build_sam


class Item:
    def __init__(self, key, value, type: str):
        self.key = key
        self.type = type
        if type.startswith("Tensor"):
            self.value = {"size": value.shape, "values": value.flatten().tolist()}
        else:
            self.value = value

    def to_dict(self):
        return {self.key: {self.type: self.value}}


def output_to_file(name: str, items: list):
    path = "~/Documents/test-outputs/" + name + ".json"
    path = os.path.expanduser(path)
    values = {}
    for item in items:
        values.update(item.to_dict())
    output = {"values": values}

    data = json.dumps(output, indent=4)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(data)


def random_tensor(shape: list, seed: int = 0):
    n = 1
    for dim in shape:
        n *= dim

    a = 3
    c = 23
    m = 2**4

    result = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        result.append(x / m)  # Normalize the result to [0, 1]
    return torch.tensor(result).view(shape)


def random_ndarray(shape: list, seed: int = 0) -> np.ndarray:
    return random_tensor(shape, seed).detach().cpu().numpy()


def build_sam_test(checkpoint: str = None):
    return _build_sam(8, 2, 2, [2, 5, 8, 11], checkpoint)


import uuid
import os
import json
from functools import reduce
import dpath
import re
from torch import nn

ignored_items = [
    "act",
    "num_heads",
    "scale",
    "use_rel_pos",
    "groups",
    "epsilon",
    "window_size",
    "eps",
    "img_size",
    "num_layers",
    "sigmoid_output",
    "num_mask_tokens",
    "skip_first_layer_pe",
    "in_channels",
    "out_channels",
    "output_upscaling2",
    "output_upscaling4",
    "padding",
    "embed_dim",
    "mask_downscaling2",
    "mask_downscaling5",
    "input_image_size",
    "image_embedding_size",
    "mask_threshold",
    "image_format",
    "num_pos_feats",
    "scale",
    "pe_layer",
]
ignored_arrays = ["stride", "kernel_size", "dilation", "padding2", "padding_out"]
ignored_three_dim_arrays = ["pixel_mean", "pixel_std"]
ignored = {
    **{key: None for key in ignored_items},
    **{key: [None, None] for key in ignored_arrays},
    **{key: [None, None, None] for key in ignored_three_dim_arrays},
}
transposed = [
    "lin1.weight",
    "lin2.weight",
    "qkv.weight",
    "proj.weight",
    "gamma.weight",
    "stride.weight",
    "padding.weight",
]

replace = [
    ["norm1.weight", "norm1.gamma"],
    ["norm2.weight", "norm2.gamma"],
    ["norm3.weight", "norm3.gamma"],
    ["norm4.weight", "norm4.gamma"],
    ["norm1.bias", "norm1.beta"],
    ["norm2.bias", "norm2.beta"],
    ["norm3.bias", "norm3.beta"],
    ["norm4.bias", "norm4.beta"],
    ["norm_final_attn.weight", "norm_final_attn.gamma"],
    ["norm_final_attn.bias", "norm_final_attn.beta"],
]

sequences = ["neck", "output_upscaling", "mask_downscaling"]


def input_to_file(file_name: str, model: nn.Module):
    data = {}
    params = model.named_parameters()
    for name, param in params:
        for item in transposed:
            if item in name and len(param.shape) == 2:
                param = param.transpose(0, 1)
                break

        # for mask_decoder mlp Linear
        if re.match(r".*layers\.\d+\.weight", name) and len(param.shape) == 2:
            param = param.transpose(0, 1)

        for item in replace:
            name = name.replace(item[0], item[1])
        for sequence in sequences:
            name = name.replace(sequence + ".", sequence)
            keys = name.split(".")

        for i in range(len(keys) + 1):
            next_key_is_number = i + 1 < len(keys) and keys[i + 1].isdigit()
            if not next_key_is_number:
                for item in ignored_items:
                    dpath.new(data, "/".join(keys[: i + 1] + [item]), None)
                for item in ignored_arrays:
                    dpath.new(data, "/".join(keys[: i + 1] + [item]), [None, None])
                for item in ignored_three_dim_arrays:
                    dpath.new(
                        data, "/".join(keys[: i + 1] + [item]), [None, None, None]
                    )
        param_id = str(uuid.uuid4())
        param_shape = list(param.size())
        param_value = (
            param.flatten().type(torch.float64).detach().cpu().numpy().tolist()
        )
        param_data = {
            "id": param_id,
            "param": {"value": param_value, "shape": param_shape},
        }

        dpath.new(data, name.replace(".", "/"), param_data)

    json_data = {
        "metadata": {
            "float": "f64",
            "int": "i64",
            "format": "burn_core::record::file::PrettyJsonFileRecorderSIMD<burn_core::record::settings::DoublePrecisionSettings>",
            "version": "0.7.0",
            "settings": "DoublePrecisionSettings",
        },
        "item": {**ignored, **data},
    }
    path = "~/Documents/sam-models/" + file_name + ".json"
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(json_data, json_file)
