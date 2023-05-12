use pyo3::{types::PyModule, Py, PyAny, PyResult, Python};

pub fn module_to_file(file: &str, py: Python, module: &PyAny) -> PyResult<()> {
    let fun: Py<PyAny> = PyModule::from_code(
        py,
        r#"
import uuid
import os
import json
from functools import reduce
import dpath
import re
import torch
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


def module_to_file(file_name: str, model: nn.Module):
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
            "float": "f32",
            "int": "f32",
            "format": "burn_core::record::file::PrettyJsonFileRecorderSIMD<burn_core::record::settings::FullPrecisionSettings>",
            "version": "0.8.0",
            "settings": "FullPrecisionSettings"
        },
        "item": {**ignored, **data},
    }
    path = "~/Documents/sam-models/" + file_name + ".json"
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(json_data, json_file)
        
    "#,
        "",
        "",
    )?
    .getattr("module_to_file")?
    .into();
    fun.call1(py, (file, module))?;
    Ok(())
}
