use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};

use crate::{burn_helpers::TensorHelpers, sam_predictor::Size};

///Multi-head Attention block with relative position embeddings.
#[derive(Debug, Module)]
pub struct Attention<B: Backend> {
    num_heads: usize,
    scale: f64,
    pub qkv: Linear<B>,
    pub proj: Linear<B>,
    use_rel_pos: bool,
    rel_pos_h: Option<Param<Tensor<B, 2>>>,
    rel_pos_w: Option<Param<Tensor<B, 2>>>,
}
impl<B: Backend> Attention<B> {
    // Args:
    // dim (int): Number of input channels.
    // num_heads (int): Number of attention heads.
    // qkv_bias (bool):  If True, add a learnable bias to query, key, value.
    // rel_pos (bool): If True, add relative positional embeddings to the attention map.
    // rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
    // input_size (tuple(int, int) or None): Input resolution for calculating the relative
    //     positional parameter size.
    pub fn new(
        dim: usize,
        num_heads: Option<usize>,
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
        let qkv = LinearConfig::new(dim, 3 * dim).with_bias(qkv_bias).init();
        let proj = LinearConfig::new(dim, dim).init();
        let mut rel_pos_h = None;
        let mut rel_pos_w = None;
        if use_rel_pos {
            assert!(
                input_size.is_some(),
                "Input size must be provided if using relative positional encoding."
            );
            let Size(h, w) = input_size.unwrap();
            rel_pos_h = Some(Param::from(Tensor::zeros([2 * h - 1, head_dim])));
            rel_pos_w = Some(Param::from(Tensor::zeros([2 * w - 1, head_dim])));
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
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = x.dims();
        let (b, h, w) = (shape[0], shape[1], shape[2]);

        let qkv = self
            .qkv
            .forward(x)
            .reshape_max([b, h * w, 3, self.num_heads, usize::MAX])
            .permute([2, 0, 3, 1, 4]);
        let qkv = qkv
            .reshape_max([3, b * self.num_heads, h * w, usize::MAX])
            .unbind(0);
        let (q, k, v) = (qkv[0].clone(), qkv[1].clone(), qkv[2].clone());

        let mut attn = (q.clone() * self.scale).matmul(k.transpose());
        if self.use_rel_pos {
            attn = add_decomposed_rel_pos(
                attn,
                q,
                self.rel_pos_h.clone().unwrap().val(),
                self.rel_pos_w.clone().unwrap().val(),
                Size(h, w),
                Size(h, w),
            )
        };
        attn = softmax(attn, 2);

        let x = attn
            .matmul(v)
            .reshape_max([b, self.num_heads, h, w, usize::MAX])
            .permute([0, 2, 3, 1, 4])
            .reshape_max([b, h, w, usize::MAX]);
        let x = self.proj.forward(x);
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
fn add_decomposed_rel_pos<B: Backend>(
    attn: Tensor<B, 3>,
    q: Tensor<B, 3>,
    rel_pos_h: Tensor<B, 2>,
    rel_pos_w: Tensor<B, 2>,
    q_size: Size,
    k_size: Size,
) -> Tensor<B, 3> {
    let Size(q_h, q_w) = q_size;
    let Size(k_h, k_w) = k_size;
    let rh = get_rel_pos(q_h, k_h, rel_pos_h);
    let rw = get_rel_pos(q_w, k_w, rel_pos_w);

    let shape = q.dims();
    let (b, dim) = (shape[0], shape[2]);
    let r_q = q.reshape([b, q_h, q_w, dim]);

    let rel_h: Tensor<B, 4> = Tensor::einsum("bhwc,hkc->bhwk", r_q.clone(), rh);
    let rel_w: Tensor<B, 4> = Tensor::einsum("bhwc,wkc->bhwk", r_q, rw);
    let attn = attn.reshape([b, q_h, q_w, k_h, k_w])
        + rel_h.unsqueeze().permute([1, 2, 3, 4, 0])
        + rel_w.unsqueeze().permute([1, 2, 3, 0, 4]);
    attn.reshape([b, q_h * q_w, k_h * k_w])
}

// Get relative positional embeddings according to the relative positions of
// query and key sizes.
// Args:
// q_size (int): size of query q.
// k_size (int): size of key k.
// rel_pos (Tensor): relative position embeddings (L, C).

// Returns:
// Extracted positional embeddings according to relative positions.
fn get_rel_pos<B: Backend>(q_size: usize, k_size: usize, rel_pos: Tensor<B, 2>) -> Tensor<B, 3> {
    let max_rel_dist = 2 * q_size.max(k_size) - 1;
    let mut rel_pos_resized = rel_pos;

    let dim = rel_pos_resized.dims()[0];
    if dim != max_rel_dist {
        rel_pos_resized = rel_pos_resized
            .reshape_max([1, dim, usize::MAX])
            .permute([0, 2, 1])
            .upsample_linear1d::<3>(&[max_rel_dist], false, None)
            .reshape_max([usize::MAX, max_rel_dist])
            .permute([1, 0]);
    }
    let q_coords = Tensor::arange(0..q_size)
        .unsqueeze()
        .mul_scalar((k_size as f64 / q_size as f64).max(1.0))
        .permute([1, 0]);
    let k_coords = Tensor::arange(0..k_size)
        .unsqueeze()
        .mul_scalar((q_size as f64 / k_size as f64).max(1.0));
    let relative_coords =
        (q_coords - k_coords) + (k_size as f64 - 1.) * (q_size as f64 / k_size as f64).max(1.0);
    let idk = rel_pos_resized.index_tch(vec![relative_coords]); // Todo 40 out of range
    idk
}

#[cfg(test)]
pub mod test {

    use crate::{
        sam_predictor::Size,
        tests::helpers::{load_module, random_tensor, Test, TestBackend},
    };

    #[test]
    fn test_get_rel_pos() {
        let rel_pos = random_tensor([127, 40], 1);
        let q_size = 32;
        let k_size = 32;
        let output = super::get_rel_pos::<TestBackend>(q_size, k_size, rel_pos.clone());
        let file = Test::open("get_rel_pos");
        file.equal("input", rel_pos);
        file.equal("output", output);
    }

    #[test]
    fn test_add_decomposed_rel_pos() {
        let attn = random_tensor([200, 49, 49], 2);
        let q = random_tensor([200, 49, 20], 3);
        let rel_pos_h = random_tensor([20, 20], 4);
        let rel_pos_w = random_tensor([20, 20], 5);
        let q_size = Size(7, 7);
        let k_size = Size(7, 7);
        let output = super::add_decomposed_rel_pos::<TestBackend>(
            attn.clone(),
            q.clone(),
            rel_pos_h,
            rel_pos_w,
            q_size,
            k_size,
        );
        let file = Test::open("add_decomposed_rel_pos");
        file.equal("attn", attn);
        file.equal("q", q);
        file.equal("q_size", q_size);
        file.equal("k_size", k_size);
        file.almost_equal("output", output, None);
    }

    #[test]
    fn test_attention() {
        let mut attention = super::Attention::<TestBackend>::new(
            320,
            Some(16),
            Some(true),
            Some(true),
            Some(true),
            Some(Size(14, 14)),
        );
        attention = load_module("attention", attention);

        // Forward
        let input = random_tensor([25, 14, 14, 320], 1);
        let output = attention.forward(input.clone());
        let file = Test::open("attention");
        file.equal("input", input);
        file.almost_equal("output", output, 0.01);
    }
}
