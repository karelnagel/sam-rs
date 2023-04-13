use super::common::{Activation, LayerNorm2d};
use crate::{modeling::common::MLPBlock, sam_predictor::Size};
use tch::nn::Path;
use tch::{
    nn::{self, ConvConfig, LinearConfig, Module},
    Device, Kind, Tensor,
};

/// This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
pub struct ImageEncoderViT {
    pub img_size: i64,
    patch_embed: PatchEmbed,
    pos_embed: Option<Tensor>,
    blocks: Vec<Block>,
    neck: nn::Sequential,
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
            .add(nn::conv2d(vs, embed_dim, out_chans, 1, Default::default()))
            .add(LayerNorm2d::new(vs, out_chans, Default::default()))
            .add(nn::conv2d(vs, out_chans, out_chans, 3, Default::default()))
            .add(LayerNorm2d::new(vs, out_chans, Default::default()));

        Self {
            img_size,
            patch_embed,
            pos_embed,
            blocks,
            neck,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = self.patch_embed.forward(x);
        x = if let Some(pos_embed) = &self.pos_embed {
            x + pos_embed
        } else {
            x
        };

        for blk in &self.blocks {
            x = blk.forward(&x);
        }

        let list = self.neck.forward_all(&x.permute(&[0, 3, 1, 2]), None);
        list[0].copy()
    }
}

///Transformer blocks with support of window attention and residual propagation blocks
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
        let x = self.norm1.forward(x);

        // Window partition
        let (mut x, pad_hw) = if self.window_size > 0 {
            let (x, pad_hw) = window_partition(x, self.window_size);
            (x, Some(pad_hw))
        } else {
            (x, None)
        };
        x = self.attn.forward(&x);
        // Reverse window partition
        x = if self.window_size > 0 {
            let (h, w) = x.size2().unwrap();
            window_unpartition(x, self.window_size, pad_hw.unwrap(), Size(h, w))
        } else {
            x
        };

        x = shortcut + x;
        x = &x + self.mlp.forward(&self.norm2.forward(&x));
        x
    }
}

///Multi-head Attention block with relative position embeddings.
pub struct Attention {
    num_heads: i64,
    scale: f64,
    qkv: nn::Linear,
    proj: nn::Linear,
    use_rel_pos: bool,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}
impl Attention {
    // Args:
    // dim (int): Number of input channels.
    // num_heads (int): Number of attention heads.
    // qkv_bias (bool):  If True, add a learnable bias to query, key, value.
    // rel_pos (bool): If True, add relative positional embeddings to the attention map.
    // rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
    // input_size (tuple(int, int) or None): Input resolution for calculating the relative
    //     positional parameter size.
    pub fn new(
        vs: &nn::Path,
        dim: i64,
        num_heads: Option<i64>,
        qkv_bias: Option<bool>,
        use_rel_pos: Option<bool>,
        _rel_pos_zero_init: Option<bool>,
        input_size: Option<Size>,
    ) -> Self {
        let num_heads = num_heads.unwrap_or(8);
        let qkv_bias = qkv_bias.unwrap_or(true);
        let use_rel_pos = use_rel_pos.unwrap_or(false);
        let _rel_pos_zero_init = _rel_pos_zero_init.unwrap_or(true);

        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);
        let qkv = nn::linear(
            vs,
            dim,
            dim * 3,
            LinearConfig {
                bias: qkv_bias,
                ..Default::default()
            },
        );
        let proj = nn::linear(vs, dim, dim, Default::default());

        let mut rel_pos_h = None;
        let mut rel_pos_w = None;
        if use_rel_pos {
            assert!(
                input_size.is_some(),
                "Input size must be provided if using relative positional encoding."
            );
            let (h, w) = (input_size.unwrap().0, input_size.unwrap().1);
            rel_pos_h = Some(vs.zeros("rel_pos_h", &[w, num_heads]));
            rel_pos_w = Some(vs.zeros("rel_pos_w", &[h, num_heads]));
        }

        Self {
            num_heads,
            scale,
            qkv,
            proj,
            use_rel_pos,
            rel_pos_h,
            rel_pos_w,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (b, h, w, _) = x.size4().unwrap();
        let qkv = self
            .qkv
            .forward(x)
            .reshape(&[b, h * w, 3, self.num_heads, -1])
            .permute(&[2, 0, 3, 1, 4]);
        let qkv = qkv.reshape(&[3, b * self.num_heads, h * w, -1]).unbind(0);
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);
        let mut attn = (q * self.scale) * k.transpose(-2, -1);
        if self.use_rel_pos {
            attn = add_decomposed_rel_pos(
                &attn,
                &q,
                &self.rel_pos_h.as_ref().unwrap(),
                &self.rel_pos_w.as_ref().unwrap(),
                Size(h, w),
                Size(h, w),
            );
        }
        attn = attn.softmax(-1, Kind::Float);
        let x = (attn * v)
            .view([b, self.num_heads, h, w, -1])
            .permute(&[0, 2, 3, 1, 4])
            .reshape(&[b, h, w, -1]);
        self.proj.forward(&x)
    }
}

// Partition into non-overlapping windows with padding if needed.
// Args:
//     x (tensor): input tokens with [B, H, W, C].
//     window_size (int): window size.

// Returns:
//     windows: windows after partition with [B * num_windows, window_size, window_size, C].
//     (Hp, Wp): padded height and width before partition
pub fn window_partition(x: Tensor, window_size: i64) -> (Tensor, Size) {
    let (b, h, w, c) = x.size4().unwrap();

    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;
    let x = if pad_h > 0 || pad_w > 0 {
        x.pad(&[0, 0, 0, pad_w, 0, pad_h], "d", Some(0.0)) // Todo probably wrong
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
pub fn window_unpartition(windows: Tensor, window_size: i64, pad_hw: Size, hw: Size) -> Tensor {
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

// Get relative positional embeddings according to the relative positions of
// query and key sizes.
// Args:
// q_size (int): size of query q.
// k_size (int): size of key k.
// rel_pos (Tensor): relative position embeddings (L, C).

// Returns:
// Extracted positional embeddings according to relative positions.
pub fn get_rel_pos(q_size: i64, k_size: i64, rel_pos: Tensor) -> Tensor {
    let max_rel_dist = 2 * q_size.max(k_size) - 1;

    // Interpolate rel pos if needed.
    let rel_pos_resized = if rel_pos.size()[0] != max_rel_dist {
        // Interpolate rel pos.
        let rel_pos_resized = rel_pos
            .reshape(&[1, rel_pos.size()[0], -1])
            .permute(&[0, 2, 1])
            // Todo
            // .interpolate(
            //     &[max_rel_dist],
            //     InterpolateConfig {
            //         mode: "linear",
            //         size: max_rel_dist,
            //         ..Default::default()
            //     },
            // )
            .reshape(&[-1, max_rel_dist])
            .permute(&[1, 0]);
        rel_pos_resized
    } else {
        rel_pos
    };
    let q_coords = Tensor::arange(q_size, (Kind::Int64, Device::Cpu)).unsqueeze(-1)
        * (k_size as f64 / q_size as f64).max(1.0);
    let k_coords = Tensor::arange(k_size, (Kind::Int64, Device::Cpu)).unsqueeze(0)
        * (q_size as f64 / k_size as f64).max(1.0);
    let relative_coords =
        &q_coords - &k_coords + (k_size - 1) as f64 * (q_size as f64 / k_size as f64).max(1.0);
    rel_pos_resized.index(&[Some(relative_coords.to_kind(tch::Kind::Int64))])
}

// Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
// https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
// Args:
//     attn (Tensor): attention map.
//     q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
//     rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
//     rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
//     q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
//     k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

// Returns:
//     attn (Tensor): attention map with added relative positional embeddings.
pub fn add_decomposed_rel_pos(
    attn: &Tensor,
    q: &Tensor,
    rel_pos_h: &Tensor,
    rel_pos_w: &Tensor,
    q_size: Size,
    k_size: Size,
) -> Tensor {
    let Size(q_h, q_w) = q_size;
    let Size(k_h, k_w) = k_size;
    let rh = get_rel_pos(q_h, k_h, rel_pos_h.copy());
    let rw = get_rel_pos(q_w, k_w, rel_pos_w.copy());

    let (b, _, dim) = q.size3().unwrap();
    let r_q = q.reshape(&[b, q_h, q_w, dim]);
    let rel_h = r_q.matmul(&rh).view([b, q_h, q_w, k_h, k_w]);
    let rel_w = r_q.matmul(&rw).view([b, q_h, q_w, k_h, k_w]);
    let attn = attn.view([b, q_h, q_w, k_h, k_w]) + &rel_h.unsqueeze(-1) + &rel_w.unsqueeze(-2);
    attn.view([b, q_h * q_w, k_h * k_w])
}

/// Image to Patch Embedding.
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
