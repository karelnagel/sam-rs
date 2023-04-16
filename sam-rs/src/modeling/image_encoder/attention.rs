use tch::{
    nn::{self, LinearConfig, Module},
    Tensor,
};

use crate::sam_predictor::Size;

///Multi-head Attention block with relative position embeddings.
#[derive(Debug)]
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
        attn = attn.softmax(-1, tch::Kind::Float);
        let x = (attn * v)
            .view([b, self.num_heads, h, w, -1])
            .permute(&[0, 2, 3, 1, 4])
            .reshape(&[b, h, w, -1]);
        self.proj.forward(&x)
    }
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
fn add_decomposed_rel_pos(
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

// Get relative positional embeddings according to the relative positions of
// query and key sizes.
// Args:
// q_size (int): size of query q.
// k_size (int): size of key k.
// rel_pos (Tensor): relative position embeddings (L, C).

// Returns:
// Extracted positional embeddings according to relative positions.
fn get_rel_pos(q_size: i64, k_size: i64, rel_pos: Tensor) -> Tensor {
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
    let q_coords = Tensor::arange(q_size, (tch::Kind::Int64, tch::Device::Cpu)).unsqueeze(-1)
        * (k_size as f64 / q_size as f64).max(1.0);
    let k_coords = Tensor::arange(k_size, (tch::Kind::Int64, tch::Device::Cpu)).unsqueeze(0)
        * (q_size as f64 / k_size as f64).max(1.0);
    let relative_coords =
        &q_coords - &k_coords + (k_size - 1) as f64 * (q_size as f64 / k_size as f64).max(1.0);
    rel_pos_resized.index(&[Some(relative_coords.to_kind(tch::Kind::Int64))])
}
