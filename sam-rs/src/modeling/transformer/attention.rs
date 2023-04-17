use tch::{
    nn::{self, Module},
    Tensor,
};

// An attention layer that allows for downscaling the size of the embedding
//     after projection to queries, keys, and values.
#[derive(Debug)]
pub struct Attention {
    embedding_dim: i64,
    internal_dim: i64,
    num_heads: i64,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
}

impl Attention {
    pub fn new(
        vs: &nn::Path,
        embedding_dim: i64,
        num_heads: i64,
        downsample_rate: Option<i64>,
    ) -> Self {
        let downsample_rate = downsample_rate.unwrap_or(1);
        let internal_dim = embedding_dim / downsample_rate;
        let q_proj = nn::linear(vs, embedding_dim, internal_dim, Default::default());
        let k_proj = nn::linear(vs, embedding_dim, internal_dim, Default::default());
        let v_proj = nn::linear(vs, embedding_dim, internal_dim, Default::default());
        let out_proj = nn::linear(vs, internal_dim, embedding_dim, Default::default());
        Self {
            embedding_dim,
            internal_dim,
            num_heads,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        }
    }

    fn _separate_heads(&self, x: &Tensor, num_heads: i64) -> Tensor {
        let (b, n, c) = x.size3().unwrap();
        let x = x.reshape(&[b, n, num_heads, c / num_heads]);
        x.transpose(1, 2)
    }

    fn _recombine_heads(&self, x: Tensor) -> Tensor {
        let (b, n_heads, n_tokens, c_per_head) = x.size4().unwrap();
        x.transpose(1, 2)
            .reshape(&[b, n_tokens, n_heads * c_per_head])
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        // # Input projections
        let q = self.q_proj.forward(q);
        let k = self.k_proj.forward(k);
        let v = self.v_proj.forward(v);

        // # Separate into heads
        let q = self._separate_heads(&q, self.num_heads);
        let k = self._separate_heads(&k, self.num_heads);
        let v = self._separate_heads(&v, self.num_heads);

        // # Attention
        let (_, _, _, c_per_head) = q.size4().unwrap();
        let mut attn = q.matmul(&k.transpose(2, 3)); // B x N_heads x N_tokens x N_tokens
        attn = attn / (c_per_head as f64).sqrt();
        attn = attn.softmax(-1, tch::Kind::Float);

        // # Get output
        let out = attn.matmul(&v);
        let out = self._recombine_heads(out);
        self.out_proj.forward(&out)
    }
}
