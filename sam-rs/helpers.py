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
    path = "test-outputs/"+name+".json"
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


def input_to_file(file_name:str,model:nn.Module):
    # Initialize the JSON data structure
    json_data = {
        "metadata": {
            "float": "f32",
            "int": "i32",
            "format": "burn_core::record::file::FilePrettyJsonRecorder",
            "version": "0.6.0",
            "settings": "DebugRecordSettings"
        },
        "item": {"act":None}
    }

    # Iterate through the model's named parameters (weights and biases)
    for name, param in model.named_parameters():
        # Extract layer name and parameter type (weight or bias)
        layer_name, param_type = name.split('.')
        
        # Create a unique ID for each parameter
        param_id = str(uuid.uuid4())

        # Convert the parameter values and shape to lists
        param_shape = list(param.size())
        param_value = param.flatten().detach().cpu().numpy().tolist()

        # Update the JSON data structure
        if layer_name not in json_data["item"]:
            json_data["item"][layer_name] = {}
        json_data["item"][layer_name][param_type] = {
            "id": param_id,
            "param": {
                "value": param_value,
                "shape": param_shape
            }
        }
    path = "test-inputs/"+file_name+'.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
