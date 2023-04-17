use tch::{
    nn::{self, Module},
    Tensor,
};

use crate::{
    modeling::common::{activation::Activation, mlp_block::MLPBlock},
    sam_predictor::Size,
};

use super::attention::Attention;

///Transformer blocks with support of window attention and residual propagation blocks
#[derive(Debug)]
pub struct Block {
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    attn: Attention,
    window_size: i64,
    mlp: MLPBlock,
}
impl Block {
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
        vs: &nn::Path,
        dim: i64,
        num_heads: i64,
        mlp_ratio: Option<f64>,
        qkv_bias: Option<bool>,
        act_layer: Activation,
        use_rel_pos: Option<bool>,
        rel_pos_zero_init: Option<bool>,
        window_size: Option<i64>,
        input_size: Option<Size>,
    ) -> Self {
        let mlp_ratio = mlp_ratio.unwrap_or(4.0);
        let qkv_bias = qkv_bias.unwrap_or(true);
        let use_rel_pos = use_rel_pos.unwrap_or(false);
        let rel_pos_zero_init = rel_pos_zero_init.unwrap_or(true);
        let window_size = window_size.unwrap_or(0);

        let norm1 = nn::layer_norm(vs, vec![dim], Default::default());
        let norm2 = nn::layer_norm(vs, vec![dim], Default::default());
        let input_size = if window_size != 0 {
            Some(Size(window_size, window_size))
        } else {
            input_size
        };
        let attn = Attention::new(
            vs,
            dim,
            Some(num_heads),
            Some(qkv_bias),
            Some(use_rel_pos),
            Some(rel_pos_zero_init),
            input_size,
        );
        let mlp = MLPBlock::new(vs, dim, dim * mlp_ratio as i64, act_layer);
        Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shortcut = x.copy();
        let mut x = self.norm1.forward(x);
        let mut pad_hw = None;
        // Window partition
        let size = Size(x.size()[1], x.size()[2]);
        if self.window_size > 0 {
            let res = window_partition(x, self.window_size);
            x = res.0;
            pad_hw = Some(res.1);
        };
        x = self.attn.forward(&x);
        // Reverse window partition
        if self.window_size > 0 {
            x = window_unpartition(x, self.window_size, pad_hw.unwrap(), size)
        };

        x = shortcut + x;
        x = &x + self.mlp.forward(&self.norm2.forward(&x));
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
fn window_partition(x: Tensor, window_size: i64) -> (Tensor, Size) {
    let (b, h, w, c) = x.size4().unwrap();

    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;
    let x = if pad_h > 0 || pad_w > 0 {
        x.pad(&[0, 0, 0, pad_w, 0, pad_h], "constant", None)
    } else {
        x
    };
    let (hp, wp) = (h + pad_h, w + pad_w);
    let x = x.view([
        b,
        hp / window_size,
        window_size,
        wp / window_size,
        window_size,
        c,
    ]);
    let windows =
        x.permute(&[0, 1, 3, 2, 4, 5])
            .contiguous()
            .view([-1, window_size, window_size, c]);
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
fn window_unpartition(windows: Tensor, window_size: i64, pad_hw: Size, hw: Size) -> Tensor {
    let Size(hp, wp) = pad_hw;
    let Size(h, w) = hw;
    let b = windows.size()[0] / (hp * wp / window_size / window_size);
    let x = windows.view([
        b,
        hp / window_size,
        wp / window_size,
        window_size,
        window_size,
        -1,
    ]);
    let x = x
        .permute(&[0, 1, 3, 2, 4, 5])
        .contiguous()
        .view([b, hp, wp, -1]);
    if hp > h || wp > w {
        x.slice(1, 0, h, 1).slice(2, 0, w, 1).contiguous()
    } else {
        x
    }
}

#[cfg(test)]
mod test {
    use crate::{
        modeling::common::activation::{Activation, ActivationType},
        sam_predictor::Size,
        tests::{
            helpers::{random_tensor, TestFile},
            mocks::Mock,
        },
    };

    impl Mock for super::Block {
        fn mock(&mut self) {
            self.norm1.mock();
            self.attn.mock();
            self.norm2.mock();
            self.mlp.mock();
        }
    }
    #[test]
    fn test_window_partition() {
        let input = random_tensor(&[2, 256, 16, 16], 1);
        let (output, pad_hw) = super::window_partition(input.copy(), 16);
        let file = TestFile::open("window_partition");
        file.compare("input", input);
        file.compare("output", output);
        file.compare("size", pad_hw);
    }

    #[test]
    fn test_window_unpartition() {
        let input = random_tensor(&[2, 256, 16, 16], 2);
        let output = super::window_unpartition(input.copy(), 16, Size(16, 16), Size(14, 14));
        let file = TestFile::open("window_unpartition");
        file.compare("input", input);
        file.compare("output", output);
    }

    #[test]
    fn test_block() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let mut block = super::Block::new(
            &vs.root(),
            320,
            16,
            Some(4.0),
            Some(true),
            Activation::new(ActivationType::GELU),
            Some(true),
            Some(true),
            Some(14),
            Some(Size(64, 64)),
        );
        let file = TestFile::open("block");
        file.compare("window_size", block.window_size);

        // Mocking
        block.mock();

        // Forward
        let input = random_tensor(&[1, 64, 64, 320], 1);
        let output = block.forward(&input);
        let file = TestFile::open("block_forward");
        file.compare("input", input);
        file.compare("output", output);
    }
}
