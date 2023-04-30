use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor},
};

use crate::{
    burn_helpers::TensorHelpers,
    modeling::common::{activation::Activation, mlp_block::MLPBlock},
    sam_predictor::Size,
};

use super::attention::Attention;

///Transformer blocks with support of window attention and residual propagation blocks
#[derive(Debug, Module)]
pub struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    attn: Attention<B>,
    window_size: usize,
    mlp: MLPBlock<B>,
}
impl<B: Backend> Block<B> {
    // Args:
    // dim (int): Number of input channels.
    // num_heads (int): Number of attention heads in each ViT block.
    // mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    // qkv_bias (bool): If True, add a learnable bias to query, key, value.
    // norm_layer (nn.Module): Normalization layer.
    // act_layer (nn.Module): Activation layer.
    // use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
    // rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
    // window_size (int): Window size for window attention blocks. If it equals 0, then
    //     use global attention.
    // input_size (tuple(int, int) or None): Input resolution for calculating the relative
    //     positional parameter size.
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: Option<f64>,
        qkv_bias: Option<bool>,
        act_layer: Activation,
        use_rel_pos: Option<bool>,
        rel_pos_zero_init: Option<bool>,
        window_size: Option<usize>,
        input_size: Option<Size>,
    ) -> Self {
        let mlp_ratio = mlp_ratio.unwrap_or(4.0);
        let qkv_bias = qkv_bias.unwrap_or(true);
        let use_rel_pos = use_rel_pos.unwrap_or(false);
        let rel_pos_zero_init = rel_pos_zero_init.unwrap_or(true);
        let window_size = window_size.unwrap_or(0);

        let norm1 = LayerNormConfig::new(dim).init();
        let norm2 = LayerNormConfig::new(dim).init();
        let input_size = if window_size != 0 {
            Some(Size(window_size, window_size))
        } else {
            input_size
        };
        let attn = Attention::new(
            dim,
            Some(num_heads),
            Some(qkv_bias),
            Some(use_rel_pos),
            Some(rel_pos_zero_init),
            input_size,
        );
        let mlp = MLPBlock::new(dim, (dim as f64 * mlp_ratio) as usize, act_layer);
        Self {
            norm1: norm1.into(),
            attn: attn.into(),
            norm2: norm2.into(),
            mlp: mlp.into(),
            window_size,
        }
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shortcut = x.clone();
        let mut x = self.norm1.forward(x);
        let mut pad_hw = None;
        // Window partition
        let shape = x.dims();
        let size = Size(shape[1], shape[2]);
        if self.window_size > 0 {
            let (res, size) = window_partition(x, self.window_size);
            x = res;
            pad_hw = Some(size);
        };
        x = self.attn.forward(x);
        // Reverse window partition
        if self.window_size > 0 {
            x = window_unpartition(x, self.window_size, pad_hw.unwrap(), size)
        };

        x = shortcut + x;
        x = x.clone() + self.mlp.forward(self.norm2.forward(x));
        x
    }
}

// Partition into non-overlapping windows with padding if needed.
// Args:
//     x (tensor): input tokens with [B, H, W, C].
//     window_size (int): window size.

// Returns:
//     windows: windows after partition with [B * num_windows, window_size, window_size, C].
//     (Hp, Wp): padded height and width before partition
fn window_partition<B: Backend>(x: Tensor<B, 4>, window_size: usize) -> (Tensor<B, 4>, Size) {
    let shape = x.dims();
    let (b, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);

    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;
    let x = if pad_h > 0 || pad_w > 0 {
        x.pad(&[0, 0, 0, pad_w, 0, pad_h], "constant", None)
    } else {
        x
    };
    let (hp, wp) = (h + pad_h, w + pad_w);
    let x = x.reshape([
        b,
        hp / window_size,
        window_size,
        wp / window_size,
        window_size,
        c,
    ]);
    let windows =
        x.permute([0, 1, 3, 2, 4, 5])
            .reshape_max([usize::MAX, window_size, window_size, c]);
    (windows, Size(hp, wp))
}

// Window unpartition into original sequences and removing padding.
//     Args:
//         windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
//         window_size (int): window size.
//         pad_hw (Tuple): padded height and width (Hp, Wp).
//         hw (Tuple): original height and width (H, W) before padding.

//     Returns:
//         x: unpartitioned sequences with [B, H, W, C].
fn window_unpartition<B: Backend>(
    windows: Tensor<B, 4>,
    window_size: usize,
    pad_hw: Size,
    hw: Size,
) -> Tensor<B, 4> {
    let Size(hp, wp) = pad_hw;
    let Size(h, w) = hw;
    let b = windows.dims()[0] / (hp * wp / window_size / window_size);
    let x = windows.reshape_max([
        b,
        hp / window_size,
        wp / window_size,
        window_size,
        window_size,
        usize::MAX,
    ]);
    let x = x
        .permute([0, 1, 3, 2, 4, 5])
        .reshape_max([b, hp, wp, usize::MAX]);
    if hp > h || wp > w {
        x.narrow(1, 0, h).narrow(2, 0, w)
    } else {
        x
    }
}

#[cfg(test)]
mod test {

    use crate::{
        modeling::common::activation::Activation,
        sam_predictor::Size,
        tests::helpers::{load_module, random_tensor, Test, TestBackend},
    };

    #[test]
    fn test_window_partition() {
        let input = random_tensor([2, 256, 16, 16], 1);
        let (output, pad_hw) = super::window_partition::<TestBackend>(input.clone(), 16);
        let file = Test::open("window_partition");
        file.compare("input", input);
        file.compare("output", output);
        file.compare("size", pad_hw);
    }

    #[test]
    fn test_window_unpartition() {
        let input = random_tensor([2, 256, 16, 16], 2);
        let output =
            super::window_unpartition::<TestBackend>(input.clone(), 16, Size(16, 16), Size(14, 14));
        let file = Test::open("window_unpartition");
        file.compare("input", input);
        file.compare("output", output);
    }

    #[test]
    fn test_block() {
        let mut block = super::Block::<TestBackend>::new(
            320,
            16,
            Some(4.0),
            Some(true),
            Activation::GELU,
            Some(true),
            Some(true),
            Some(14),
            Some(Size(64, 64)),
        );
        block = load_module("block", block);// Todo

        // Forward
        let input = random_tensor([1, 64, 64, 320], 1);
        let output = block.forward(input.clone());
        let file = Test::open("block");
        file.compare("input", input);
        file.compare("output", output);
    }
}
