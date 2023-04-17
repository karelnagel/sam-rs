use tch::{
    nn::{self, Module},
    Tensor,
};

use crate::modeling::common::{mlp_block::MLPBlock, activation::{Activation, ActivationType}};

use super::attention::Attention;

#[derive(Debug)]
pub struct TwoWayAttentionBlock {
    self_attn: Attention,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
    norm4: nn::LayerNorm,
    cross_attn_token_to_image: Attention,
    cross_attn_image_to_token: Attention,
    mlp: MLPBlock,
    skip_first_layer_pe: bool,
}
impl TwoWayAttentionBlock {
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
        vs: &nn::Path,
        embedding_dim: i64,
        num_heads: i64,
        mlp_dim: Option<i64>,
        activation: Option<Activation>,
        attention_downsample_rate: Option<i64>,
        skip_first_layer_pe: Option<bool>,
    ) -> Self {
        let mlp_dim = mlp_dim.unwrap_or(2048);
        let activation = activation.unwrap_or(Activation::new(ActivationType::ReLU));
        let attention_downsample_rate = attention_downsample_rate.unwrap_or(2);
        let skip_first_layer_pe = skip_first_layer_pe.unwrap_or(false);

        let self_attn = Attention::new(vs, embedding_dim, num_heads, None);
        let norm1 = nn::layer_norm(vs, vec![embedding_dim], Default::default());
        let cross_attn_token_to_image = Attention::new(
            vs,
            embedding_dim,
            num_heads,
            Some(attention_downsample_rate),
        );
        let norm2 = nn::layer_norm(vs, vec![embedding_dim], Default::default());
        let mlp = MLPBlock::new(vs, embedding_dim, mlp_dim, activation);
        let norm3 = nn::layer_norm(vs, vec![embedding_dim], Default::default());
        let norm4 = nn::layer_norm(vs, vec![embedding_dim], Default::default());
        let cross_attn_image_to_token = Attention::new(
            vs,
            embedding_dim,
            num_heads,
            Some(attention_downsample_rate),
        );
        Self {
            self_attn,
            norm1,
            norm2,
            norm3,
            norm4,
            mlp,
            cross_attn_token_to_image,
            cross_attn_image_to_token,
            skip_first_layer_pe,
        }
    }

    pub fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        query_pe: &Tensor,
        key_pe: &Tensor,
    ) -> (Tensor, Tensor) {
        let mut queries = queries.copy();
        let keys = keys.copy();
        // Self attention block
        if self.skip_first_layer_pe {
            queries = self.self_attn.forward(&queries, &queries, &queries);
        } else {
            let q = &queries + query_pe;
            let attn_out = self.self_attn.forward(&q, &q, &queries);
            queries = &queries + attn_out;
        }
        queries = self.norm1.forward(&queries);

        // Cross attention block, tokens attending to image embedding
        let q = &queries + query_pe;
        let k = &keys + key_pe;
        let attn_out = self.cross_attn_token_to_image.forward(&q, &k, &keys);
        queries = &queries + attn_out;
        queries = self.norm2.forward(&queries);

        // MLP block
        let mlp_out = self.mlp.forward(&queries);
        queries = &queries + mlp_out;
        queries = self.norm3.forward(&queries);

        // Cross attention block, image attending to tokens
        let q = &queries + query_pe;
        let k = &keys + key_pe;
        let attn_out = self.cross_attn_image_to_token.forward(&q, &k, &queries);
        queries = &queries + attn_out;
        queries = self.norm4.forward(&queries);

        (queries, keys)
    }
}
