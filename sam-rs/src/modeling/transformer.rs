use tch::{
    nn::{self, Module},
    Tensor,
};

use super::common::{Activation, ActivationType, MLPBlock};

pub struct TwoWayTransformer {
    depth: i64,
    embedding_dim: i64,
    num_heads: i64,
    mlp_dim: i64,
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
            depth,
            embedding_dim,
            num_heads,
            mlp_dim,
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
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> (Tensor, Tensor) {
        // BxCxHxW -> BxHWxC == B x N_image_tokens x C
        let (bs, c, h, w) = image_embedding.size4().unwrap();
        let image_embedding = image_embedding.flatten(2, 3).permute(&[0, 2, 1]);
        let image_pe = image_pe.flatten(2, 3).permute(&[0, 2, 1]);

        //  Prepare queries
        let mut queries = point_embedding.copy();
        let mut keys = image_embedding;
        for layer in &self.layers {
            let (q, k) = layer.forward(&queries, &keys, &point_embedding, &image_pe);
            queries = q.copy();
            keys = k.copy();
        }
        let q = &queries + point_embedding;
        let k = &keys + image_pe;
        let attn_out = &self.final_attn_token_to_image.forward(&q, &k, &keys);
        queries = &queries + attn_out;
        queries = self.norm_final_attn.forward(&queries);
        (queries, keys)
    }
}

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

// An attention layer that allows for downscaling the size of the embedding
//     after projection to queries, keys, and values.
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

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
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
