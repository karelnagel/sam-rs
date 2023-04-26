use self::block::Block;
use self::patch_embed::PatchEmbed;

use super::common::{activation::Activation, layer_norm_2d::LayerNorm2d};
use crate::sam_predictor::Size;
use burn::{
    module::{Module, Param},
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{backend::Backend, Tensor},
};
mod attention;
mod block;
mod patch_embed;

/// This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
#[derive(Debug, Module)]
pub struct ImageEncoderViT<B: Backend> {
    pub img_size: usize,
    patch_embed: PatchEmbed<B>,
    pos_embed: Option<Param<Tensor<B, 4>>>,
    blocks: Vec<Block<B>>,
    neck1: Conv2d<B>,
    neck2: LayerNorm2d<B>,
    neck3: Conv2d<B>,
    neck4: LayerNorm2d<B>,
    _embed_dim: usize, //Needed for mocking
    _out_chans: usize, // Needed for mocking
}
impl<B: Backend> ImageEncoderViT<B> {
    // Args:
    //         img_size (int): Input image size.
    //         patch_size (int): Patch size.
    //         in_chans (int): Number of input image channels.
    //         embed_dim (int): Patch embedding dimension.
    //         depth (int): Depth of ViT.
    //         num_heads (int): Number of attention heads in each ViT block.
    //         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    //         qkv_bias (bool): If True, add a learnable bias to query, key, value.
    //         norm_layer (nn.Module): Normalization layer.
    //         act_layer (nn.Module): Activation layer.
    //         use_abs_pos (bool): If True, use absolute positional embeddings.
    //         use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
    //         rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
    //         window_size (int): Window size for window attention blocks.
    //         global_attn_indexes (list): Indexes for blocks using global attention.
    pub fn new(
        img_size: Option<usize>,
        patch_size: Option<usize>,
        in_chans: Option<usize>,
        embed_dim: Option<usize>,
        depth: Option<usize>,
        num_heads: Option<usize>,
        mlp_ratio: Option<f64>,
        out_chans: Option<usize>,
        qkv_bias: Option<bool>,
        act_layer: Activation,
        use_abs_pos: Option<bool>,
        use_rel_pos: Option<bool>,
        rel_pos_zero_init: Option<bool>,
        window_size: Option<usize>,
        global_attn_indexes: Option<Vec<usize>>,
    ) -> Self {
        let img_size = img_size.unwrap_or(1024);
        let patch_size = patch_size.unwrap_or(16);
        let in_chans = in_chans.unwrap_or(3);
        let embed_dim = embed_dim.unwrap_or(768);
        let depth = depth.unwrap_or(12);
        let num_heads = num_heads.unwrap_or(12);
        let mlp_ratio = mlp_ratio.unwrap_or(4.0);
        let out_chans = out_chans.unwrap_or(256);
        let qkv_bias = qkv_bias.unwrap_or(true);
        let use_abs_pos = use_abs_pos.unwrap_or(true);
        let use_rel_pos = use_rel_pos.unwrap_or(false);
        let rel_pos_zero_init = rel_pos_zero_init.unwrap_or(true);
        let window_size = window_size.unwrap_or(0);
        let global_attn_indexes = global_attn_indexes.unwrap_or(vec![]);

        let patch_embed = PatchEmbed::new(
            Some(Size(patch_size, patch_size)),
            Some(Size(patch_size, patch_size)),
            None,
            Some(in_chans),
            Some(embed_dim),
        );
        let mut pos_embed = None;
        if use_abs_pos {
            pos_embed = Some(Param::from(Tensor::zeros([
                1,
                img_size / patch_size,
                img_size / patch_size,
                embed_dim,
            ])));
        }

        let mut blocks = vec![];

        for i in 0..depth {
            let window_size = if !global_attn_indexes.contains(&i) {
                window_size
            } else {
                0
            };
            let block = Block::new(
                embed_dim,
                num_heads,
                Some(mlp_ratio),
                Some(qkv_bias),
                act_layer,
                Some(use_rel_pos),
                Some(rel_pos_zero_init),
                Some(window_size),
                // window_size=window_size if i not in global_attn_indexes else 0, // Todo
                Some(Size(img_size / patch_size, img_size / patch_size)),
            );
            blocks.push(block);
        }

        let neck1 = Conv2dConfig::new([embed_dim, out_chans], [1, 1]).init();
        let neck2 = LayerNorm2d::new(out_chans, Default::default());
        let neck3 = Conv2dConfig::new([out_chans, out_chans], [3, 3])
            .with_bias(false)
            .init();
        let neck4 = LayerNorm2d::new(out_chans, Default::default());
        Self {
            img_size,
            patch_embed: patch_embed,
            pos_embed: pos_embed,
            blocks,
            neck1,
            neck2,
            neck3,
            neck4,
            _embed_dim: embed_dim,
            _out_chans: out_chans,
        }
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.patch_embed.forward(x);
        if let Some(pos_embed) = &self.pos_embed {
            x = x + pos_embed.val()
        }
        for blk in &self.blocks {
            x = blk.forward(x);
        }
        self.neck(x.permute([0, 3, 1, 2]))
    }
    fn neck(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.neck1.forward(x);
        x = self.neck2.forward(x);
        x = self.neck3.forward(x);
        x = self.neck4.forward(x);
        x
    }
}

#[cfg(test)]
mod test {
    use super::ImageEncoderViT;
    use crate::{
        modeling::common::activation::Activation,
        tests::helpers::{random_tensor, Test, TestBackend},
    };

    #[test]
    fn test_image_encoder() {
        let act = Activation::GELU;
        let img_size = 32;
        let image_encoder = ImageEncoderViT::<TestBackend>::new(
            Some(img_size),
            Some(4),
            Some(3),
            Some(80),
            Some(32),
            Some(16),
            Some(4.0),
            Some(256),
            Some(true),
            act,
            Some(true),
            Some(true),
            Some(true),
            Some(14),
            Some(vec![7, 15, 23, 31]),
        );
        let file = Test::open("image_encoder");
        file.compare("img_size", img_size);

        // Forward
        let input = random_tensor([1, 3, img_size, img_size], 1);
        let output = image_encoder.forward(input.clone());
        let file = Test::open("image_encoder_forward");
        file.compare("input", input);
        file.compare("output", output);
    }
}
