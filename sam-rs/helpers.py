import numpy as np
import json
import os
import json
import os
import torch
import torch
import numpy as np
from torch import nn
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.sam import Sam
from segment_anything.modeling.image_encoder import PatchEmbed
from segment_anything.modeling.image_encoder import Attention
from segment_anything.modeling.image_encoder import Block
from segment_anything.modeling.transformer import Attention
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.mask_decoder import MLP
from segment_anything.modeling.transformer import TwoWayAttentionBlock
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
from segment_anything.modeling.prompt_encoder import PromptEncoder

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


def to_file(name:str,items:list):
    path = "test-files/"+name+".json"
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
    m = 2**32
    
    result = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        result.append(x / m)  # Normalize the result to [0, 1]
    return torch.tensor(result).view(shape)

def random_ndarray(shape:list,seed:int=0)->np.ndarray:
    return random_tensor(shape,seed).detach().cpu().numpy()

# Mocking 
def mock_linear(linear:nn.Linear )->nn.Linear:
    linear.weight.data = random_tensor(linear.weight.size(),1)
    linear.bias.data = random_tensor(linear.bias.size(),2)

def mock_layer_norm(layer_norm: nn.LayerNorm)->nn.LayerNorm:
    layer_norm.weight.data = random_tensor(layer_norm.weight.size(),1)
    layer_norm.bias.data = random_tensor(layer_norm.bias.size(),2)

def mock_conv2d(conv2d:nn.Conv2d)->nn.Conv2d:
    conv2d.weight.data = random_tensor(conv2d.weight.size(),1)
    if conv2d.bias is not None:
        conv2d.bias.data = random_tensor(conv2d.bias.size(),2)

def mock_embedding(embedding:nn.Embedding)->nn.Embedding:
    embedding.weight.data = random_tensor(embedding.weight.size(),1)

def mock_conv_transpose2d(conv: nn.ConvTranspose2d)->nn.ConvTranspose2d:
    conv.weight.data = random_tensor(conv.weight.size(),1)
    conv.bias.data = random_tensor(conv.bias.size(),2)

def mock_tensor(tensor:torch.Tensor)->torch.Tensor:
    tensor.data = random_tensor(tensor.size(),1)

def mock_mlp_block(mlp_block:MLPBlock)->MLPBlock:
    mock_linear(mlp_block.lin1)
    mock_linear(mlp_block.lin2)

def mock_patch_embed(patch_embed:PatchEmbed)->PatchEmbed:
    mock_conv2d(patch_embed.proj)

def mock_attention(attention:Attention)->Attention:
    mock_linear(attention.qkv)
    mock_linear(attention.proj)

def mock_block(block:Block)->Block:
    mock_layer_norm(block.norm1)
    mock_layer_norm(block.norm2)
    mock_attention(block.attn)
    mock_mlp_block(block.mlp )


def mock_image_encoder(image_encoder:ImageEncoderViT)->ImageEncoderViT:
    mock_patch_embed(image_encoder.patch_embed)
    for block in image_encoder.blocks:
        mock_block(block)
    shape =image_encoder.neck[0].weight.shape
    embed_dim = shape[1]
    out_chans = shape[0]
    print(embed_dim,out_chans)
    conv1 =nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            )
    conv2 = nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            )
    mock_conv2d(conv1)
    mock_conv2d(conv2)
    image_encoder.neck = nn.Sequential(
            conv1,
            LayerNorm2d(out_chans),
            conv2,
            LayerNorm2d(out_chans),
        )

def mock_transformer_attention(attention:Attention)->Attention:
    mock_linear(attention.q_proj)
    mock_linear(attention.k_proj)
    mock_linear(attention.v_proj)
    mock_linear(attention.out_proj)

def mock_transformer_two_way_attention_block(block:TwoWayAttentionBlock)->TwoWayAttentionBlock:
    mock_layer_norm(block.norm1)
    mock_layer_norm(block.norm2)
    mock_layer_norm(block.norm3)
    mock_layer_norm(block.norm4)
    mock_transformer_attention(block.cross_attn_image_to_token)
    mock_transformer_attention(block.cross_attn_token_to_image)
    mock_transformer_attention(block.self_attn)
    mock_mlp_block(block.mlp)

def mock_transformer_two_way_transformer(transformer:TwoWayTransformer)->TwoWayTransformer:
    for i in range(len(transformer.layers)):
        mock_transformer_two_way_attention_block(transformer.layers[i])
    mock_transformer_attention(transformer.final_attn_token_to_image)
    mock_layer_norm(transformer.norm_final_attn)

def mock_mlp(mlp:MLP)->MLP:
    for i in range(len(mlp.layers)):
        mock_linear(mlp.layers[i])


def mock_mask_decoder(mask_decoder:MaskDecoder)->MaskDecoder:
    mock_transformer_two_way_transformer(mask_decoder.transformer)
    mock_embedding(mask_decoder.iou_token)
    mock_embedding(mask_decoder.mask_tokens)
    for i in range(len(mask_decoder.output_hypernetworks_mlps)):
        mock_mlp(mask_decoder.output_hypernetworks_mlps[i])
    mock_mlp(mask_decoder.iou_prediction_head)
    conv = nn.ConvTranspose2d(mask_decoder.transformer_dim, mask_decoder.transformer_dim // 4, kernel_size=2, stride=2)
    conv2 = nn.ConvTranspose2d(mask_decoder.transformer_dim // 4, mask_decoder.transformer_dim // 8, kernel_size=2, stride=2)
    mock_conv_transpose2d(conv)
    mock_conv_transpose2d(conv2)
    mask_decoder.output_upscaling = nn.Sequential(
            conv,
            LayerNorm2d(mask_decoder.transformer_dim // 4),
            nn.GELU(),
            conv2,
            nn.GELU(),
        )
    
def mock_position_embedding_random(position_embedding:PositionEmbeddingRandom)->PositionEmbeddingRandom:
    mock_tensor(position_embedding.positional_encoding_gaussian_matrix)

def mock_prompt_encoder(encoder:PromptEncoder)->PromptEncoder:
    mock_position_embedding_random(encoder.pe_layer)
    for i in range(len(encoder.point_embeddings)):
        mock_embedding(encoder.point_embeddings[i])
    mock_embedding(encoder.no_mask_embed)
    mock_embedding(encoder.not_a_point_embed)
    shape =encoder.mask_downscaling[6].weight.shape
    mask_in_chans = shape[1]
    embed_dim = shape[0]
    conv1 = nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2)
    conv2 = nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2)
    conv3 = nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
    mock_conv2d(conv1)
    mock_conv2d(conv2)
    mock_conv2d(conv3)
    encoder.mask_downscaling = nn.Sequential(
            conv1,
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            conv2,
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            conv3,
        )
    

def mock_sam(sam:Sam):
    mock_prompt_encoder(sam.prompt_encoder)
    mock_mask_decoder(sam.mask_decoder)
    mock_image_encoder(sam.image_encoder)