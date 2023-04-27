import numpy as np
import json
import os
import json
import os
import torch
import torch
import numpy as np
from torch import nn
import uuid
from segment_anything.build_sam import _build_sam

class Item:
    def __init__(self, key, value, type:str):
        self.key = key
        self.type = type
        if type.startswith("Tensor"):
            self.value = {"size":value.shape,"values":value.flatten().tolist()}
        else:
            self.value = value

    def to_dict(self):
        return {self.key: {self.type: self.value}}


def output_to_file(name:str,items:list):
    path = "~/Documents/test-outputs/"+name+".json"
    path = os.path.expanduser(path)
    values = {}
    for item in items:
        values.update(item.to_dict())
    output = {"values": values}
    
    data = json.dumps(output, indent=4)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(data)

def random_tensor(shape:list,seed:int=0):
    n = 1 
    for dim in shape:
        n*=dim

    a = 3
    c = 23
    m = 2**4
    
    result = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        result.append(x / m)  # Normalize the result to [0, 1]
    return torch.tensor(result).view(shape)

def random_ndarray(shape:list,seed:int=0)->np.ndarray:
    return random_tensor(shape,seed).detach().cpu().numpy()

from functools import reduce

def set_nested_key(dct, keys, value):
    reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], dct)[keys[-1]] = value

def input_to_file(file_name: str, model: nn.Module):
    json_data = {
        "metadata": {
            "float": "f32",
            "int": "i32",
            "format": "burn_core::record::file::FilePrettyJsonRecorder",
            "version": "0.6.0",
            "settings": "DebugRecordSettings"
        },
        "item": {"act": None}
    }
    for name, param in model.named_parameters():
        keys = name.split('.')
        param_id = str(uuid.uuid4())
        param_shape = list(param.size())
        param_value = param.flatten().detach().cpu().numpy().tolist()
        param_data = {
            "id": param_id,
            "param": {
                "value": param_value,
                "shape": param_shape
            }
        }
        set_nested_key(json_data["item"], keys, param_data)
    
    path = "~/Documents/test-inputs/" + file_name + '.json'
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

def build_sam_test(checkpoint:str=None):
    return _build_sam(64,4,4,[2,5,8,11],checkpoint)