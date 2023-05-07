import torch
from helpers import build_sam_test, input_to_file
from segment_anything.build_sam import build_sam_vit_h, build_sam_vit_l,build_sam_vit_b
import sys

def convert(type:str, file:str):
    # for test it creates one with random values and also saves it in .pth
    print("Loading weights from "+file+".pth")
    if type =="test":
        sam = build_sam_test(None)
        torch.save(sam.state_dict(),file+".pth")
    elif type == "vit_h":
        sam = build_sam_vit_h(file+".pth")
    elif type == "vit_l":
        sam = build_sam_vit_l(file+".pth")
    elif type == "vit_b":
        sam = build_sam_vit_b(file+".pth")
    else:
        raise Exception("Unknown type: "+type)
    print("Saving input to "+file+".json")
    input_to_file(file,sam)
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: python convert.py <test|vit_h|vit_l|vit_b> <file>")
    type = sys.argv[1]
    file = sys.argv[2]
    convert(type,file)