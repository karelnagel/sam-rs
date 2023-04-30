use crate::sam_predictor::Size;
use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::Conv2dPaddingConfig;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Image to Patch Embedding.
#[derive(Debug, Module)]
pub struct PatchEmbed<B: Backend> {
    proj: Conv2d<B>,
}
impl<B: Backend> PatchEmbed<B> {
    // Args:
    //         kernel_size (Tuple): kernel size of the projection layer.
    //         stride (Tuple): stride of the projection layer.
    //         padding (Tuple): padding size of the projection layer.
    //         in_chans (int): Number of input image channels.
    //         embed_dim (int):  embed_dim (int): Patch embedding dimension.
    pub fn new(
        kernel_size: Option<Size>,
        stride: Option<Size>,
        padding: Option<Size>,
        in_chans: Option<usize>,
        embed_dim: Option<usize>,
    ) -> Self {
        let kernel_size = kernel_size.unwrap_or(Size(16, 16));
        let stride = stride.unwrap_or(Size(16, 16));
        let padding = padding.unwrap_or(Size(0, 0));
        let in_chans = in_chans.unwrap_or(3);
        let embed_dim = embed_dim.unwrap_or(768);
        let proj = Conv2dConfig::new([in_chans, embed_dim], [kernel_size.0, kernel_size.1])
            .with_stride([stride.0, stride.1])
            .with_padding(Conv2dPaddingConfig::Explicit(padding.0, padding.1))
            .init();
        Self { proj: proj.into() }
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.proj.forward(x);
        x.permute([0, 2, 3, 1])
    }
}

#[cfg(test)]
mod test {
    use crate::{
        sam_predictor::Size,
        tests::helpers::{load_module, random_tensor, Test, TestBackend},
    };

    use super::PatchEmbed;

    #[test]
    fn test_patch_embed() {
        let mut patch_embed = PatchEmbed::<TestBackend>::new(
            Some(Size(16, 16)),
            Some(Size(16, 16)),
            Some(Size(0, 0)),
            Some(3),
            Some(320),
        );
        patch_embed = load_module("patch_embed", patch_embed);

        // Forward
        let input = random_tensor([1, 3, 512, 512], 3);
        let output = patch_embed.forward(input.clone());
        let file = Test::open("patch_embed");
        file.compare("input", input);
        file.compare("output", output);
    }
}
