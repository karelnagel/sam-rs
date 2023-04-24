use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};

// An attention layer that allows for downscaling the size of the embedding
//     after projection to queries, keys, and values.
#[derive(Debug, Module)]
pub struct Attention<B: Backend> {
    _embedding_dim: usize,
    _internal_dim: usize,
    num_heads: usize,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
}
impl<B: Backend> Attention<B> {
    pub fn new(embedding_dim: usize, num_heads: usize, downsample_rate: Option<usize>) -> Self {
        let downsample_rate = downsample_rate.unwrap_or(1);
        let internal_dim = embedding_dim / downsample_rate;

        let q_proj = LinearConfig::new(embedding_dim, internal_dim).init();
        let k_proj = LinearConfig::new(embedding_dim, internal_dim).init();
        let v_proj = LinearConfig::new(embedding_dim, internal_dim).init();
        let out_proj = LinearConfig::new(internal_dim, embedding_dim).init();
        Self {
            _embedding_dim: embedding_dim,
            _internal_dim: internal_dim,
            num_heads,
            q_proj: q_proj.into(),
            k_proj: k_proj.into(),
            v_proj: v_proj.into(),
            out_proj: out_proj.into(),
        }
    }

    fn _separate_heads(&self, x: Tensor<B, 3>, num_heads: usize) -> Tensor<B, 4> {
        let shape = x.shape().dims;
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let x = x.reshape([b, n, num_heads, c / num_heads]);
        // x.transpose(1, 2)
        x.transpose()
    }

    fn _recombine_heads(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let shape = x.shape().dims;
        let (b, n_heads, n_tokens, c_per_head) = (shape[0], shape[1], shape[2], shape[3]);
        x.transpose()
            // x.transpose(1, 2)
            .reshape([b, n_tokens, n_heads * c_per_head])
    }

    pub fn forward(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        // # Input projections
        let q = self.q_proj.forward(q);
        let k = self.k_proj.forward(k);
        let v = self.v_proj.forward(v);

        // # Separate into heads
        let q = self._separate_heads(q, self.num_heads);
        let k = self._separate_heads(k, self.num_heads);
        let v = self._separate_heads(v, self.num_heads);

        // # Attention
        let c_per_head = q.shape().dims[3];
        let mut attn = q.matmul(k.transpose());
        // let mut attn = q.matmul(&k.transpose(2, 3)); // B x N_heads x N_tokens x N_tokens
        attn = attn / (c_per_head as f64).sqrt();
        attn = softmax(attn, usize::MAX);

        // # Get output
        let out = attn.matmul(v);
        let out = self._recombine_heads(out);
        self.out_proj.forward(out)
    }
}

#[cfg(test)]
mod test {
    use crate::tests::helpers::{random_tensor, Test, TestBackend};

    #[test]
    fn test_attention() {
        let mut attention = super::Attention::<TestBackend>::new(256, 8, Some(1));
        let file = Test::open("transformer_attention");
        file.compare("embedding_dim", attention._embedding_dim);
        file.compare("internal_dim", attention._internal_dim);
        file.compare("num_heads", attention.num_heads);
        // file.compare("q_proj_size", attention.q_proj.ws.size());
        // file.compare("k_proj_size", attention.k_proj.ws.size());
        // file.compare("v_proj_size", attention.v_proj.ws.size());
        // file.compare("out_proj_size", attention.out_proj.ws.size());

        // Forward
        let q = random_tensor([1, 256, 256], 1);
        let k = random_tensor([1, 256, 256], 2);
        let v = random_tensor([1, 256, 256], 3);
        let output = attention.forward(q.clone(), k.clone(), v.clone());
        let file = Test::open("transformer_attention_forward");
        file.compare("q", q);
        file.compare("k", k);
        file.compare("v", v);
        file.compare("output", output);
    }
}
