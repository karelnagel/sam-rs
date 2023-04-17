use tch::{
    nn::{self, ConvConfig, Module},
    Tensor,
};

use crate::sam_predictor::Size;

/// Image to Patch Embedding.
#[derive(Debug)]
pub struct PatchEmbed {
    proj: nn::Conv2D,
}
impl PatchEmbed {
    // Args:
    //         kernel_size (Tuple): kernel size of the projection layer.
    //         stride (Tuple): stride of the projection layer.
    //         padding (Tuple): padding size of the projection layer.
    //         in_chans (int): Number of input image channels.
    //         embed_dim (int):  embed_dim (int): Patch embedding dimension.
    pub fn new(
        vs: &nn::Path,
        kernel_size: Option<Size>,
        stride: Option<Size>,
        padding: Option<Size>,
        in_chans: Option<i64>,
        embed_dim: Option<i64>,
    ) -> Self {
        let kernel_size = kernel_size.unwrap_or(Size(16, 16));
        let stride = stride.unwrap_or(Size(16, 16));
        let padding = padding.unwrap_or(Size(0, 0));
        let in_chans = in_chans.unwrap_or(3);
        let embed_dim = embed_dim.unwrap_or(768);
        let proj = nn::conv2d(
            vs,
            in_chans,
            embed_dim,
            kernel_size.0,
            ConvConfig {
                stride: stride.0,
                padding: padding.0,
                ..Default::default()
            },
        );
        Self { proj }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.proj.forward(x);
        x.permute(&[0, 2, 3, 1])
    }
}

#[cfg(test)]
mod test {
    use crate::{
        sam_predictor::Size,
        tests::helpers::{random_tensor, TestFile, ToTest},
    };
    use tch::{nn, Device};

    use super::PatchEmbed;

    #[test]
    fn test_patch_embed() {
        let vs = nn::VarStore::new(Device::Cpu);
        let mut patch_embed = PatchEmbed::new(
            &vs.root(),
            Some(Size(16, 16)),
            Some(Size(16, 16)),
            Some(Size(0, 0)),
            Some(3),
            Some(768),
        );
        patch_embed.proj.ws = random_tensor(&[2, 256, 16, 16], 1);
        patch_embed.proj.bs = Some(random_tensor(&[2], 2));
        let file = TestFile::open("patch_embed");
        file.compare("weight", &patch_embed.proj.ws.to_test());
        let bias = patch_embed.proj.bs.as_ref().unwrap();
        file.compare("bias", &bias.to_test());

        // Forward
        let input = random_tensor(&[2, 256, 16, 16], 3);
        let output = &patch_embed.forward(&input);
        let file = TestFile::open("patch_embed_forward");
        file.compare("input", &input.to_test());
        file.compare("output", &output.to_test());
    }
}
