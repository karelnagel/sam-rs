use crate::modeling::image_encoder::helpers::add_decomposed_rel_pos;
use tch::nn::ModuleT;
use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct Attention {
    num_heads: i64,
    scale: f64,
    qkv: tch::nn::Linear,
    proj: tch::nn::Linear,
    use_rel_pos: bool,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}

impl Attention {
    pub fn new(
        dim: i64,
        num_heads: i64,
        qkv_bias: bool,
        use_rel_pos: bool,
        rel_pos_zero_init: bool,
        input_size: Option<(i64, i64)>,
    ) -> Self {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);
        let qkv = tch::nn::linear(vs.root(), dim, dim * 3, qkv_bias);
        let proj = tch::nn::linear(vs.root(), dim, dim, qkv_bias);

        let (rel_pos_h, rel_pos_w) = if use_rel_pos {
            assert!(
                input_size.is_some(),
                "Input size must be provided if using relative positional encoding."
            );
            let (h, w) = input_size.unwrap();
            (
                Some(Tensor::zeros(
                    &[2 * h - 1, head_dim],
                    (Kind::Float, tch::Device::Cpu),
                )),
                Some(Tensor::zeros(
                    &[2 * w - 1, head_dim],
                    (Kind::Float, tch::Device::Cpu),
                )),
            )
        } else {
            (None, None)
        };

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
}

impl ModuleT for Attention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let (b, h, w, _) = xs.size4().unwrap();
        let qkv = self
            .qkv
            .forward_t(xs, train)
            .view((b, h * w, 3, self.num_heads, -1))
            .permute(&[2, 0, 3, 1, 4]);
        let (q, k, v) = qkv.view((3, b * self.num_heads, h * w, -1)).unbind(0);

        let mut attn = q.mul_scalar(self.scale).matmul(&k.transpose(-2, -1));

        if self.use_rel_pos {
            attn = &add_decomposed_rel_pos(
                attn,
                q,
                &self.rel_pos_h.as_ref().unwrap(),
                &self.rel_pos_w.as_ref().unwrap(),
                (h, w),
                (h, w),
            );
        }

        attn = &attn.softmax(-1, Kind::Float);
        let x = attn
            .matmul(&v)
            .view((b, self.num_heads, h, w, -1))
            .permute(&[0, 2, 3, 1, 4])
            .reshape((b, h, w, -1));
        let x = self.proj.forward_t(&x, train);

        x
    }
}
