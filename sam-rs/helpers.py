import numpy as np
import json
import os
import json
import os
import torch
import torch
import numpy as np
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

def build_sam_test(checkpoint:str=None):
    return _build_sam(64,4,4,[2,5,8,11],checkpoint)