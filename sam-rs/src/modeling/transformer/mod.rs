mod attention;
mod two_way_attention;
use tch::{
    nn::{self, Module},
    Tensor,
};

use self::{attention::Attention, two_way_attention::TwoWayAttentionBlock};

use super::common::activation::{Activation, ActivationType};

#[derive(Debug)]
pub struct TwoWayTransformer {
    _depth: i64,
    _embedding_dim: i64,
    _num_heads: i64,
    _mlp_dim: i64,
    layers: Vec<TwoWayAttentionBlock>,
    final_attn_token_to_image: Attention,
    norm_final_attn: nn::LayerNorm,
}
impl TwoWayTransformer {
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
        vs: &nn::Path,
        depth: i64,
        embedding_dim: i64,
        num_heads: i64,
        mlp_dim: i64,
        activation: Option<Activation>,
        attention_downsample_rate: Option<i64>,
    ) -> Self {
        let activation = activation.unwrap_or(Activation::new(ActivationType::ReLU));
        let attention_downsample_rate = attention_downsample_rate.unwrap_or(2);
        let mut layers: Vec<TwoWayAttentionBlock> = vec![];
        for i in 0..depth {
            layers.push(TwoWayAttentionBlock::new(
                vs,
                embedding_dim,
                num_heads,
                Some(mlp_dim),
                Some(activation),
                Some(attention_downsample_rate),
                Some(i == 0),
            ));
        }
        let final_attn_token_to_image = Attention::new(
            vs,
            embedding_dim,
            num_heads,
            Some(attention_downsample_rate),
        );
        let norm_final_attn = nn::layer_norm(vs, vec![embedding_dim], Default::default());
        Self {
            _depth: depth,
            _embedding_dim: embedding_dim,
            _num_heads: num_heads,
            _mlp_dim: mlp_dim,
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
        image_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> (Tensor, Tensor) {
        // BxCxHxW -> BxHWxC == B x N_image_tokens x C
        let image_embedding = image_embedding.flatten(2, -1).permute(&[0, 2, 1]);
        let image_pe = image_pe.flatten(2, -1).permute(&[0, 2, 1]);

        //  Prepare queries
        let mut queries = point_embedding.copy();
        let mut keys = image_embedding;
        for layer in &self.layers {
            (queries, keys) = layer.forward(&queries, &keys, &point_embedding, &image_pe);
        }
        let q = &queries + point_embedding;
        let k = &keys + image_pe;
        let attn_out = &self.final_attn_token_to_image.forward(&q, &k, &keys);
        queries = &queries + attn_out;
        queries = self.norm_final_attn.forward(&queries);
        (queries, keys)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        modeling::common::activation::{Activation, ActivationType},
        tests::{
            helpers::{random_tensor, TestFile},
            mocks::Mock,
        },
    };

    use super::TwoWayTransformer;
    impl Mock for TwoWayTransformer {
        fn mock(&mut self) {
            self.final_attn_token_to_image.mock();
            self.norm_final_attn.mock();
            for layer in &mut self.layers {
                layer.mock();
            }
        }
    }
    #[test]
    fn test_two_way_transformer() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let mut transformer = super::TwoWayTransformer::new(
            &vs.root(),
            2,
            64,
            4,
            256,
            Some(Activation::new(ActivationType::ReLU)),
            Some(2),
        );
        let file = TestFile::open("transformer_two_way_transformer");
        file.compare("depth", transformer._depth);
        file.compare("embedding_dim", transformer._embedding_dim);
        file.compare("num_heads", transformer._num_heads);
        file.compare("mlp_dim", transformer._mlp_dim);
        file.compare("layers_len", transformer.layers.len());

        // Mocking
        transformer.mock();

        // Forward
        let image_embedding = random_tensor(&[1, 64, 16, 16], 1);
        let image_pe = random_tensor(&[1, 64, 16, 16], 2);
        let point_embedding = random_tensor(&[16, 256, 64], 3);
        let (queries, keys) = transformer.forward(&image_embedding, &image_pe, &point_embedding);
        let file = TestFile::open("transformer_two_way_transformer_forward");
        file.compare("image_embedding", image_embedding);
        file.compare("image_pe", image_pe);
        file.compare("point_embedding", point_embedding);
        file.compare("queries", queries);
        file.compare("keys", keys);
    }
}
