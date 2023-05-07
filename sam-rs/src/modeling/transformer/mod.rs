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
    use crate::{
        modeling::common::activation::Activation,
        tests::helpers::{load_module, random_tensor, Test, TestBackend},
    };
    #[test]
    fn test_two_way_transformer() {
        let mut transformer = super::TwoWayTransformer::<TestBackend>::new(
            2,
            64,
            4,
            256,
            Some(Activation::ReLU),
            Some(2),
        );
        transformer = load_module("transformer_two_way_transformer", transformer);

        // Forward
        let image_embedding = random_tensor([1, 64, 16, 16], 1);
        let image_pe = random_tensor([1, 64, 16, 16], 2);
        let point_embedding = random_tensor([16, 256, 64], 3);
        let (queries, keys) = transformer.forward(
            image_embedding.clone(),
            image_pe.clone(),
            point_embedding.clone(),
        );
        let file = Test::open("transformer_two_way_transformer");
        file.equal("image_embedding", image_embedding);
        file.equal("image_pe", image_pe);
        file.almost_equal("point_embedding", point_embedding,0.001);
        file.almost_equal("queries", queries,0.002);
        file.almost_equal("keys", keys,0.001);
    }
}
