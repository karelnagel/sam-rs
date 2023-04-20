use self::block::Block;
use self::patch_embed::PatchEmbed;

use super::common::{activation::Activation, layer_norm_2d::LayerNorm2d};
use crate::sam_predictor::Size;
use tch::nn::{ConvConfig, Module, Path};
use tch::{nn, Tensor};
mod attention;
mod block;
mod patch_embed;

/// This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
#[derive(Debug)]
pub struct ImageEncoderViT {
    pub img_size: i64,
    patch_embed: PatchEmbed,
    pos_embed: Option<Tensor>,
    blocks: Vec<Block>,
    neck: nn::Sequential,
    _embed_dim: i64, //Needed for mocking
    _out_chans: i64, // Needed for mocking
}
impl ImageEncoderViT {
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
        vs: &Path,
        img_size: Option<i64>,
        patch_size: Option<i64>,
        in_chans: Option<i64>,
        embed_dim: Option<i64>,
        depth: Option<i64>,
        num_heads: Option<i64>,
        mlp_ratio: Option<f64>,
        out_chans: Option<i64>,
        qkv_bias: Option<bool>,
        act_layer: Activation,
        use_abs_pos: Option<bool>,
        use_rel_pos: Option<bool>,
        rel_pos_zero_init: Option<bool>,
        window_size: Option<i64>,
        global_attn_indexes: Option<&[i64]>,
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
        let global_attn_indexes = global_attn_indexes.unwrap_or(&[]);

        let patch_embed = PatchEmbed::new(
            vs,
            Some(Size(patch_size, patch_size)),
            Some(Size(patch_size, patch_size)),
            None,
            Some(in_chans),
            Some(embed_dim),
        );
        let mut pos_embed = None;
        if use_abs_pos {
            pos_embed = Some(vs.zeros(
                "pos_embed",
                &[1, img_size / patch_size, img_size / patch_size, embed_dim],
            ));
        }

        let mut blocks = vec![];

        for i in 0..depth {
            let window_size = if !global_attn_indexes.contains(&i) {
                window_size
            } else {
                0
            };
            let block = Block::new(
                vs,
                embed_dim,
                num_heads,
                Some(mlp_ratio),
                Some(qkv_bias),
                act_layer,
                Some(use_rel_pos),
                Some(rel_pos_zero_init),
                Some(window_size),
                // window_size=window_size if i not in global_attn_indexes else 0,
                Some(Size(img_size / patch_size, img_size / patch_size)),
            );
            blocks.push(block);
        }

        let neck = nn::seq()
            .add(nn::conv2d(
                vs,
                embed_dim,
                out_chans,
                1,
                ConvConfig {
                    bias: false,
                    ..Default::default()
                },
            ))
            .add(LayerNorm2d::new(vs, out_chans, Default::default()))
            .add(nn::conv2d(
                vs,
                out_chans,
                out_chans,
                3,
                ConvConfig {
                    bias: false,
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add(LayerNorm2d::new(vs, out_chans, Default::default()));

        Self {
            img_size,
            patch_embed,
            pos_embed,
            blocks,
            neck,
            _embed_dim: embed_dim,
            _out_chans: out_chans,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = self.patch_embed.forward(x);
        if let Some(pos_embed) = &self.pos_embed {
            x = x + pos_embed
        }
        for blk in &self.blocks {
            x = blk.forward(&x);
        }
        self.neck.forward(&x.permute(&[0, 3, 1, 2]))
    }
}

#[cfg(test)]
mod test {
    use tch::nn::{self, ConvConfig};

    use super::ImageEncoderViT;
    use crate::{
        modeling::common::{activation::Activation, layer_norm_2d::LayerNorm2d},
        tests::{
            helpers::{random_tensor, TestFile},
            mocks::Mock,
        },
    };

    impl Mock for ImageEncoderViT {
        fn mock(&mut self) {
            self.patch_embed.mock();
            self.blocks.iter_mut().for_each(|b| b.mock());
            let vs = nn::VarStore::new(tch::Device::Cpu);
            dbg!(self._embed_dim, self._out_chans);
            let embed_dim = self._embed_dim;
            let out_chans = self._out_chans;
            let mut conv1 = nn::conv2d(
                &vs.root(),
                embed_dim,
                out_chans,
                1,
                ConvConfig {
                    bias: false,
                    ..Default::default()
                },
            );
            let mut conv2 = nn::conv2d(
                &vs.root(),
                out_chans,
                out_chans,
                3,
                ConvConfig {
                    bias: false,
                    padding: 1,
                    ..Default::default()
                },
            );
            conv1.mock();
            conv2.mock();
            self.neck = nn::seq()
                .add(conv1)
                .add(LayerNorm2d::new(&vs.root(), out_chans, Default::default()))
                .add(conv2)
                .add(LayerNorm2d::new(&vs.root(), out_chans, Default::default()));
        }
    }
    #[test]
    fn test_image_encoder() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let act = Activation::new(crate::modeling::common::activation::ActivationType::GELU);
        let img_size = 128;
        let mut image_encoder = ImageEncoderViT::new(
            &vs.root(),
            Some(img_size),
            Some(4),
            Some(3),
            Some(320),
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
            Some(&[7, 15, 23, 31]),
        );
        image_encoder.mock();
        let file = TestFile::open("image_encoder");
        file.compare("img_size", img_size);

        // Forward
        let input = random_tensor(&[1, 3, img_size, img_size], 1);
        let output = image_encoder.forward(&input);
        let file = TestFile::open("image_encoder_forward");
        file.compare("input", input);
        file.compare("output", output);
    }
}
