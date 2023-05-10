mod attention;
mod two_way_attention;
use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor},
};

use self::{attention::Attention, two_way_attention::TwoWayAttentionBlock};

use super::common::activation::Activation;

#[derive(Debug, Module)]
pub struct TwoWayTransformer<B: Backend> {
    layers: Vec<TwoWayAttentionBlock<B>>,
    final_attn_token_to_image: Attention<B>,
    norm_final_attn: LayerNorm<B>,
}
impl<B: Backend> TwoWayTransformer<B> {
    // A transformer decoder that attends to an input image using
    //     queries whose positional embedding is supplied.

    //     Args:
    //       depth (int): number of layers in the transformer
    //       embedding_dim (int): the channel dimension for the input embeddings
    //       num_heads (int): the number of heads for multihead attention. Must
    //         divide embedding_dim
    //       mlp_dim (int): the channel dimension internal to the MLP block
    //       activation (nn.Module): the activation to use in the MLP block
    pub fn new(
        depth: usize,
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        activation: Option<Activation>,
        attention_downsample_rate: Option<usize>,
    ) -> Self {
        let activation = activation.unwrap_or(Activation::ReLU);
        let attention_downsample_rate = attention_downsample_rate.unwrap_or(2);

        let mut layers: Vec<TwoWayAttentionBlock<B>> = vec![];
        for i in 0..depth {
            layers.push(TwoWayAttentionBlock::new(
                embedding_dim,
                num_heads,
                Some(mlp_dim),
                Some(activation),
                Some(attention_downsample_rate),
                Some(i == 0),
            ));
        }
        let final_attn_token_to_image =
            Attention::new(embedding_dim, num_heads, Some(attention_downsample_rate));
        let norm_final_attn = LayerNormConfig::new(embedding_dim).init();
        Self {
            layers,
            final_attn_token_to_image,
            norm_final_attn,
        }
    }

    //     Args:
    //     image_embedding (torch.Tensor): image to attend to. Should be shape
    //       B x embedding_dim x h x w for any h and w.
    //     image_pe (torch.Tensor): the positional encoding to add to the image. Must
    //       have the same shape as image_embedding.
    //     point_embedding (torch.Tensor): the embedding to add to the query points.
    //       Must have shape B x N_points x embedding_dim for any N_points.

    //   Returns:
    //     torch.Tensor: the processed point_embedding
    //     torch.Tensor: the processed image_embedding
    pub fn forward(
        &self,
        image_embedding: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        point_embedding: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // BxCxHxW -> BxHWxC == B x N_image_tokens x C
        let image_embedding = image_embedding.flatten::<3>(2, 3).permute([0, 2, 1]);
        let image_pe = image_pe.flatten::<3>(2, 3).permute([0, 2, 1]);

        //  Prepare queries
        let mut queries = point_embedding.clone();
        let mut keys = image_embedding;

        // Apply transformer blocks and final layernorm
        for layer in &self.layers {
            (queries, keys) =
                layer.forward(queries, keys, point_embedding.clone(), image_pe.clone());
        }
        // Todo after this queries is NaN

        // Apply the final attention layer from the points to the image
        let q = queries.clone() + point_embedding;
        let k = keys.clone() + image_pe;
        let attn_out = self.final_attn_token_to_image.forward(q, k, keys.clone());
        queries = queries + attn_out;
        queries = self.norm_final_attn.forward(queries);
        (queries, keys)
    }
}

#[cfg(test)]
mod test {
    use pyo3::{PyResult, Python};

    use crate::{
        modeling::common::activation::Activation,
        python::{
            module_to_file::module_to_file,
            python_data::{random_python_tensor, PythonData},
        },
        tests::helpers::{load_module, TestBackend},
    };
    #[test]
    fn test_two_way_transformer() {
        const FILE: &str = "transformer_two_way_transformer";
        fn python() -> PyResult<(
            PythonData<4>,
            PythonData<4>,
            PythonData<3>,
            PythonData<3>,
            PythonData<3>,
        )> {
            Python::with_gil(|py| {
                let relu = py.import("torch.nn")?.getattr("ReLU")?;
                let module = py
                    .import("segment_anything.modeling.transformer")?
                    .getattr("TwoWayTransformer")?;
                let module = module.call1((2, 64, 4, 256, relu, 2))?;
                module_to_file(FILE, py, &module)?;

                let image_embedding = random_python_tensor(py, [1, 64, 16, 16])?;
                let image_pe = random_python_tensor(py, [1, 64, 16, 16])?;
                let point_embedding = random_python_tensor(py, [16, 256, 64])?;
                let output = module.call1((image_embedding, image_pe, point_embedding))?;
                let queries = output.get_item(0)?;
                let keys = output.get_item(1)?;
                Ok((
                    image_embedding.try_into()?,
                    image_pe.try_into()?,
                    point_embedding.try_into()?,
                    queries.try_into()?,
                    keys.try_into()?,
                ))
            })
        }
        let (image_embedding, image_pe, point_embedding, queries, keys) = python().unwrap();
        let mut transformer = super::TwoWayTransformer::<TestBackend>::new(
            2,
            64,
            4,
            256,
            Some(Activation::ReLU),
            Some(2),
        );
        transformer = load_module(FILE, transformer);

        // Forward
        let (queries2, keys2) = transformer.forward(
            image_embedding.into(),
            image_pe.into(),
            point_embedding.into(),
        );
        queries.almost_equal(queries2, 5.);
        keys.almost_equal(keys2, 5.);
    }
}
