use tch::nn::Conv2D;
use tch::nn::{Module, ModuleT};
use tch::Tensor;

use crate::modeling::common::LayerNorm2d;
use crate::modeling::image_encoder::block::Block;
use crate::modeling::image_encoder::patch_embed::PatchEmbed;

pub mod attention;
pub mod block;
pub mod helpers;
pub mod patch_embed;

#[derive(Debug)]
pub struct ImageEncoderViT {
    img_size: i64,
    patch_embed: PatchEmbed,
    pos_embed: Option<Tensor>,
    blocks: tch::nn::ModuleList,
    neck: tch::nn::Sequential,
}

impl ImageEncoderViT {
    pub fn new(
        img_size: i64,
        patch_size: i64,
        in_chans: i64,
        embed_dim: i64,
        depth: i64,
        num_heads: i64,
        mlp_ratio: f64,
        out_chans: i64,
        qkv_bias: bool,
        use_abs_pos: bool,
        use_rel_pos: bool,
        rel_pos_zero_init: bool,
        window_size: i64,
        global_attn_indexes: Vec<i64>,
    ) -> Self {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let patch_embed = PatchEmbed::new(
            &vs.root(),
            (patch_size, patch_size),
            (0, 0),
            in_chans,
            embed_dim,
        );

        let pos_embed = if use_abs_pos {
            Some(vs.root().zeros(
                "pos_embed",
                &[1, img_size / patch_size, img_size / patch_size, embed_dim],
            ))
        } else {
            None
        };

        let mut blocks = tch::nn::ModuleList::new(vs.root());
        for i in 0..depth {
            let block = Block::new(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias,
                use_rel_pos,
                rel_pos_zero_init,
                if global_attn_indexes.contains(&i) {
                    0
                } else {
                    window_size
                },
                Some((img_size / patch_size, img_size / patch_size)),
            );
            blocks.push(block);
        }

        let neck = tch::nn::seq()
            .add(Conv2D::new(
                &vs.root(),
                embed_dim,
                out_chans,
                1,
                0,
                1,
                1,
                false,
            ))
            .add_fn(|xs| LayerNorm2d::new(out_chans).forward(xs))
            .add(Conv2D::new(
                &vs.root(),
                out_chans,
                out_chans,
                3,
                1,
                1,
                1,
                false,
            ))
            .add_fn(|xs| LayerNorm2d::new(out_chans).forward(xs));

        Self {
            img_size,
            patch_embed,
            pos_embed,
            blocks,
            neck,
        }
    }
}

impl ModuleT for ImageEncoderViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut x = self.patch_embed.forward_t(xs, train);
        if let Some(pos_embed) = &self.pos_embed {
            x += pos_embed;
        }

        for block in self.blocks.iter() {
            x = block.forward_t(&x, train);
        }

        x = self.neck.forward_t(&x.permute(&[0, 3, 1, 2]), train);
        x
    }
}
