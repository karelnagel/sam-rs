use self::block::Block;
use self::patch_embed::PatchEmbed;

use super::common::{activation::Activation, layer_norm_2d::LayerNorm2d};
use crate::sam_predictor::Size;
use burn::{
    module::{Module, Param},
    nn::conv::{Conv2d, Conv2dConfig, Conv2dPaddingConfig},
    tensor::{backend::Backend, Float, Tensor},
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
    neck0: Conv2d<B>,
    neck1: LayerNorm2d<B>,
    neck2: Conv2d<B>,
    neck3: LayerNorm2d<B>,
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
            let window_size = match global_attn_indexes.contains(&i) {
                true => 0,
                false => window_size,
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
                Some(Size(img_size / patch_size, img_size / patch_size)),
            );
            blocks.push(block);
        }

        let neck0 = Conv2dConfig::new([embed_dim, out_chans], [1, 1])
            .with_bias(false)
            .init();
        let neck1 = LayerNorm2d::new(out_chans, None);
        let neck2 = Conv2dConfig::new([out_chans, out_chans], [3, 3])
            .with_bias(false)
            .with_padding(Conv2dPaddingConfig::Explicit(1, 1))
            .init();
        let neck3 = LayerNorm2d::new(out_chans, None);
        Self {
            img_size,
            patch_embed,
            pos_embed,
            blocks,
            neck0,
            neck1,
            neck2,
            neck3,
        }
    }
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
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
        let mut x = self.neck0.forward(x);
        x = self.neck1.forward(x);
        x = self.neck2.forward(x);
        x = self.neck3.forward(x);
        x
    }
}

#[cfg(test)]
mod test {
    use pyo3::{types::PyDict, PyResult, Python};

    use super::ImageEncoderViT;
    use crate::{
        modeling::common::activation::Activation,
        python::{
            module_to_file::module_to_file,
            python_data::{random_python_tensor, PythonData},
        },
        tests::helpers::{load_module, TestBackend},
    };

    #[test]
    fn test_image_encoder() {
        const FILE: &str = "image_encoder";
        fn python() -> PyResult<(PythonData<4>, PythonData<4>)> {
            Python::with_gil(|py| {
                let module = py
                    .import("segment_anything.modeling.image_encoder")?
                    .getattr("ImageEncoderViT")?;
                let layer_norm = py.import("torch.nn")?.getattr("LayerNorm")?;
                let gelu = py.import("torch.nn")?.getattr("GELU")?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("img_size", 4)?;
                kwargs.set_item("patch_size", 4)?;
                kwargs.set_item("in_chans", 3)?;
                kwargs.set_item("embed_dim", 80)?;
                kwargs.set_item("depth", 4)?;
                kwargs.set_item("num_heads", 16)?;
                kwargs.set_item("mlp_ratio", 4.0)?;
                kwargs.set_item("out_chans", 32)?;
                kwargs.set_item("qkv_bias", true)?;
                kwargs.set_item("norm_layer", layer_norm)?;
                kwargs.set_item("act_layer", gelu)?;
                kwargs.set_item("use_abs_pos", true)?;
                kwargs.set_item("use_rel_pos", true)?;
                kwargs.set_item("rel_pos_zero_init", true)?;
                kwargs.set_item("window_size", 14)?;
                kwargs.set_item("global_attn_indexes", (7, 15, 23, 31))?;
                let module = module.call((), Some(kwargs))?;
                module_to_file(FILE, py, &module)?;

                let input = random_python_tensor(py, [1, 3, 4, 4])?;
                let output = module.call1((input,))?;
                Ok((input.try_into()?, output.try_into()?))
            })
        }
        let (input, python) = python().unwrap();
        let img_size = 4;
        let mut image_encoder = ImageEncoderViT::<TestBackend>::new(
            Some(img_size),
            Some(4),
            Some(3),
            Some(80),
            Some(4),
            Some(16),
            Some(4.0),
            Some(32),
            Some(true),
            Activation::GELU,
            Some(true),
            Some(true),
            Some(true),
            Some(14),
            Some(vec![7, 15, 23, 31]),
        );
        image_encoder = load_module(FILE, image_encoder);

        // Forward
        let output = image_encoder.forward(input.into());
        python.almost_equal(output, None);
    }
}
