use tch::{
    nn::{self, LinearConfig, Module},
    Device, Kind, Tensor,
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
            let Size(h, w) = input_size.unwrap();
            rel_pos_h = Some(Tensor::zeros(
                &[2 * h - 1, head_dim],
                (tch::Kind::Float, Device::Cpu),
            ));
            rel_pos_w = Some(Tensor::zeros(
                &[2 * w - 1, head_dim],
                (tch::Kind::Float, Device::Cpu),
            ));
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

        let mut attn = (q * self.scale).matmul(&k.transpose(-2, -1));
        if self.use_rel_pos {
            attn = add_decomposed_rel_pos(
                &attn,
                &q,
                &self.rel_pos_h.as_ref().unwrap(),
                &self.rel_pos_w.as_ref().unwrap(),
                Size(h, w),
                Size(h, w),
            )
        };
        attn = attn.softmax(-1, Kind::Float);

        let x = attn
            .matmul(&v)
            .view([b, self.num_heads, h, w, -1])
            .permute(&[0, 2, 3, 1, 4])
            .reshape(&[b, h, w, -1]);
        let x = self.proj.forward(&x);
        x
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

    let rel_h = Tensor::einsum("bhwc,hkc->bhwk", &[&r_q, &rh], None);
    let rel_w = Tensor::einsum("bhwc,wkc->bhwk", &[&r_q, &rw], None);
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
    let mut rel_pos_resized = rel_pos;

    if rel_pos_resized.size()[0] != max_rel_dist {
        let rel_pos_3d = rel_pos_resized.unsqueeze(0);
        // Perform linear interpolation (upsampling) along the last dimension.
        rel_pos_resized = rel_pos_3d.upsample_linear1d(&[max_rel_dist as i64], false, None);

        // Remove the extra dimension.
        rel_pos_resized = rel_pos_resized.squeeze_dim(0);
        rel_pos_resized = rel_pos_resized
            .reshape(&[-1, max_rel_dist])
            .permute(&[1, 0])
    }
    let q_coords = Tensor::arange(q_size, (tch::Kind::Int64, tch::Device::Cpu)).unsqueeze(-1)
        * (k_size as f64 / q_size as f64).max(1.0);
    let k_coords = Tensor::arange(k_size, (tch::Kind::Int64, tch::Device::Cpu)).unsqueeze(0)
        * (q_size as f64 / k_size as f64).max(1.0);
    let relative_coords =
        &q_coords - &k_coords + (k_size - 1) as f64 * (q_size as f64 / k_size as f64).max(1.0);
    rel_pos_resized.index(&[Some(relative_coords.to_kind(tch::Kind::Int64))])
}

#[cfg(test)]
mod test {
    use crate::{
        sam_predictor::Size,
        test_helpers::{random_tensor, TestFile, ToTest},
    };

    #[test]
    fn test_get_rel_pos() {
        let rel_pos = random_tensor(&[127, 80], 1);
        let q_size = 64;
        let k_size = 64;
        let output = super::get_rel_pos(q_size, k_size, rel_pos.copy());
        let file = TestFile::open("get_rel_pos");
        file.compare("input", &rel_pos.to_test());
        file.compare("output", &output.to_test());
    }

    #[test]
    fn test_add_decomposed_rel_pos() {
        let attn = random_tensor(&[400, 196, 196], 2);
        let q = random_tensor(&[400, 196, 80], 3);
        let rel_pos_h = random_tensor(&[27, 80], 4);
        let rel_pos_w = random_tensor(&[27, 80], 5);
        let q_size = Size(14, 14);
        let k_size = Size(14, 14);
        let output =
            super::add_decomposed_rel_pos(&attn, &q, &rel_pos_h, &rel_pos_w, q_size, k_size);
        let file = TestFile::open("add_decomposed_rel_pos");
        file.compare("attn", &attn.to_test());
        file.compare("q", &q.to_test());
        file.compare("q_size", &q_size.to_test());
        file.compare("k_size", &k_size.to_test());
        file.compare("output", &output.to_test());
    }

    #[test]
    fn test_attention() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let mut attention = super::Attention::new(
            &vs.root(),
            1280,
            Some(16),
            Some(true),
            Some(true),
            Some(true),
            Some(Size(14, 14)),
        );
        let file = TestFile::open("attention");
        file.compare("num_heads", &attention.num_heads.to_test());
        file.compare("scale", &attention.scale.to_test());
        file.compare("use_rel_pos", &attention.use_rel_pos.to_test());

        let input = random_tensor(&[25, 14, 14, 1280], 1);
        attention.qkv.ws = random_tensor(&[3840, 1280], 2);
        attention.qkv.bs = Some(random_tensor(&[3840], 3));
        attention.proj.ws = random_tensor(&[1280, 1280], 4);
        attention.proj.bs = Some(random_tensor(&[1280], 5));
        let output = attention.forward(&input);
        let file = TestFile::open("attention_forward");
        file.compare("input", &input.to_test());
        file.compare("output", &output.to_test());
    }
}
