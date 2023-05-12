use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};

// An attention layer that allows for downscaling the size of the embedding
//     after projection to queries, keys, and values.
#[derive(Debug, Module)]
pub struct Attention<B: Backend> {
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
            num_heads,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        }
    }

    fn _separate_heads(&self, x: Tensor<B, 3>, num_heads: usize) -> Tensor<B, 4> {
        let shape = x.dims();
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let x = x.reshape([b, n, num_heads, c / num_heads]);
        x.swap_dims(1, 2)
    }

    fn _recombine_heads(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let shape = x.dims();
        let (b, n_heads, n_tokens, c_per_head) = (shape[0], shape[1], shape[2], shape[3]);
        x.swap_dims(1, 2)
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
        let c_per_head = q.dims()[3];
        let mut attn = q.matmul(k.permute([0, 1, 3, 2]));
        // let mut attn = q.matmul(&k.transpose(2, 3)); // B x N_heads x N_tokens x N_tokens
        attn = attn / (c_per_head as f32).sqrt();
        attn = softmax(attn, 3);

        // # Get output
        let out = attn.matmul(v);
        let out = self._recombine_heads(out);
        self.out_proj.forward(out)
    }
}

#[cfg(test)]
mod test {
    use pyo3::{PyResult, Python};

    use crate::{
        python::{
            module_to_file::module_to_file,
            python_data::{random_python_tensor, PythonData},
        },
        tests::helpers::{load_module, TestBackend},
    };

    #[test]
    fn test_attention() {
        const FILE: &str = "transformer_attention";
        fn python() -> PyResult<(PythonData<3>, PythonData<3>, PythonData<3>, PythonData<3>)> {
            Python::with_gil(|py| {
                let module = py
                    .import("segment_anything.modeling.transformer")?
                    .getattr("Attention")?;
                let module = module.call1((32, 8, 1))?;
                module_to_file(FILE, py, &module)?;

                let q = random_python_tensor(py, [1, 32, 32])?;
                let k = random_python_tensor(py, [1, 32, 32])?;
                let v = random_python_tensor(py, [1, 32, 32])?;
                let output = module.call1((q, k, v))?;
                Ok((
                    q.try_into()?,
                    k.try_into()?,
                    v.try_into()?,
                    output.try_into()?,
                ))
            })
        }
        let (q, k, v, python) = python().unwrap();
        let mut attention = super::Attention::<TestBackend>::new(32, 8, Some(1));
        attention = load_module(FILE, attention);

        let output = attention.forward(q.into(), k.into(), v.into());
        python.almost_equal(output, 5.);
    }
}
