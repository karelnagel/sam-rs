use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor},
};

use crate::modeling::common::{activation::Activation, mlp_block::MLPBlock};

use super::attention::Attention;

#[derive(Debug, Module)]
pub struct TwoWayAttentionBlock<B: Backend> {
    self_attn: Attention<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,
    norm4: LayerNorm<B>,
    cross_attn_token_to_image: Attention<B>,
    cross_attn_image_to_token: Attention<B>,
    mlp: MLPBlock<B>,
    skip_first_layer_pe: bool,
}
impl<B: Backend> TwoWayAttentionBlock<B> {
    // A transformer block with four layers: (1) self-attention of sparse
    // inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
    // block on sparse inputs, and (4) cross attention of dense inputs to sparse
    // inputs.

    // Arguments:
    //   embedding_dim (int): the channel dimension of the embeddings
    //   num_heads (int): the number of heads in the attention layers
    //   mlp_dim (int): the hidden dimension of the mlp block
    //   activation (nn.Module): the activation of the mlp block
    //   skip_first_layer_pe (bool): skip the PE on the first layer
    pub fn new(
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: Option<usize>,
        activation: Option<Activation>,
        attention_downsample_rate: Option<usize>,
        skip_first_layer_pe: Option<bool>,
    ) -> Self {
        let mlp_dim = mlp_dim.unwrap_or(2048);
        let activation = activation.unwrap_or(Activation::ReLU);
        let attention_downsample_rate = attention_downsample_rate.unwrap_or(2);
        let skip_first_layer_pe = skip_first_layer_pe.unwrap_or(false);

        let self_attn = Attention::new(embedding_dim, num_heads, None);
        let norm1 = LayerNormConfig::new(embedding_dim).init();
        let cross_attn_token_to_image =
            Attention::new(embedding_dim, num_heads, Some(attention_downsample_rate));
        let norm2 = LayerNormConfig::new(embedding_dim).init();
        let mlp = MLPBlock::new(embedding_dim, mlp_dim, activation);
        let norm3 = LayerNormConfig::new(embedding_dim).init();
        let norm4 = LayerNormConfig::new(embedding_dim).init();
        let cross_attn_image_to_token =
            Attention::new(embedding_dim, num_heads, Some(attention_downsample_rate));
        Self {
            self_attn,
            norm1,
            norm2,
            norm3,
            norm4,
            cross_attn_token_to_image,
            cross_attn_image_to_token,
            mlp,
            skip_first_layer_pe,
        }
    }

    pub fn forward(
        &self,
        queries: Tensor<B, 3>,
        keys: Tensor<B, 3>,
        query_pe: Tensor<B, 3>,
        key_pe: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut queries = queries;
        let mut keys = keys;

        // Self attention block
        if self.skip_first_layer_pe {
            queries = self
                .self_attn
                .forward(queries.clone(), queries.clone(), queries.clone());
        } else {
            let q = queries.clone() + query_pe.clone();
            let attn_out = self
                .self_attn
                .forward(q.clone(), q.clone(), queries.clone());
            queries = queries + attn_out;
        }
        queries = self.norm1.forward(queries);

        // Cross attention block, tokens attending to image embedding
        let q = queries.clone() + query_pe.clone();
        let k = keys.clone() + key_pe.clone();
        let attn_out = self.cross_attn_token_to_image.forward(q, k, keys.clone());
        queries = queries + attn_out;
        queries = self.norm2.forward(queries);

        // MLP block
        let mlp_out = self.mlp.forward(queries.clone());
        queries = queries + mlp_out;
        queries = self.norm3.forward(queries);

        // Cross attention block, image attending to tokens

        let q = queries.clone() + query_pe;
        let k = keys.clone() + key_pe;
        let attn_out = self
            .cross_attn_image_to_token
            .forward(k, q, queries.clone());

        keys = keys + attn_out;
        keys = self.norm4.forward(keys);

        (queries, keys)
    }
}

#[cfg(test)]
mod test {
    use pyo3::{types::PyTuple, PyResult, Python};

    use crate::{
        modeling::common::activation::Activation,
        python::{
            module_to_file::module_to_file,
            python_data::{random_python_tensor, PythonData},
        },
        tests::helpers::{load_module, TestBackend},
    };

    #[test]
    fn test_two_way_attention_block() {
        const FILE: &str = "transformer_two_way_attention_block";
        fn python() -> PyResult<(
            PythonData<3>,
            PythonData<3>,
            PythonData<3>,
            PythonData<3>,
            PythonData<3>,
            PythonData<3>,
        )> {
            Python::with_gil(|py| {
                let relu = py.import("torch.nn")?.getattr("ReLU")?;
                let module = py
                    .import("segment_anything.modeling.transformer")?
                    .getattr("TwoWayAttentionBlock")?;
                let module = module.call1((256, 8, 2048, relu, 2, false))?;
                module_to_file(FILE, py, &module)?;

                let queries = random_python_tensor(py, [1, 256, 256]);
                let keys = random_python_tensor(py, [1, 256, 256]);
                let query_pe = random_python_tensor(py, [1, 256, 256]);
                let key_pe = random_python_tensor(py, [1, 256, 256]);
                let output = module
                    .call1((queries, keys, query_pe, key_pe))?
                    .downcast::<PyTuple>()?;
                let out_queries = output.get_item(0)?;
                let out_keys = output.get_item(1)?;
                Ok((
                    queries.into(),
                    keys.into(),
                    query_pe.into(),
                    key_pe.into(),
                    out_queries.into(),
                    out_keys.into(),
                ))
            })
        }
        let (queries, keys, query_pe, key_pe, python1, python2) = python().unwrap();
        let mut block = super::TwoWayAttentionBlock::<TestBackend>::new(
            256,
            8,
            Some(2048),
            Some(Activation::ReLU),
            Some(2),
            Some(false),
        );
        block = load_module(FILE, block);

        let (out_queries, out_keys) =
            block.forward(queries.into(), keys.into(), query_pe.into(), key_pe.into());
        python1.almost_equal(out_queries, 0.5);
        python2.almost_equal(out_keys, 0.1);
    }
}
