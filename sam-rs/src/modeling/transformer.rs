use tch::{
    nn::{self, Module},
    Tensor,
};

use crate::modeling::mask_decoder::Activation;

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
        depth: i64,
        embedding_dim: i64,
        num_heads: i64,
        mlp_dim: i64,
        activation: Option<Activation>,
        attention_downsample_rate: Option<i64>,
    ) -> Self {
        let activation = activation.unwrap_or(Activation::ReLU);
        let attention_downsample_rate = attention_downsample_rate.unwrap_or(2);
        let mut layers: Vec<TwoWayAttentionBlock> = vec![];
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
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let norm_final_attn = nn::layer_norm(&vs.root(), vec![embedding_dim], Default::default());
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

pub struct TwoWayAttentionBlock {}
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
        embedding_dim: i64,
        num_heads: i64,
        mlp_dim: Option<i64>,
        activation: Option<Activation>,
        attention_downsample_rate: Option<i64>,
        skip_first_layer_pe: Option<bool>,
    ) -> Self {
        let mlp_dim = mlp_dim.unwrap_or(2048);
        let activation = activation.unwrap_or(Activation::ReLU);
        let attention_downsample_rate = attention_downsample_rate.unwrap_or(2);
        let skip_first_layer_pe = skip_first_layer_pe.unwrap_or(false);
        unimplemented!()
        // super().__init__()
        // self.self_attn = Attention(embedding_dim, num_heads)
        // self.norm1 = nn.LayerNorm(embedding_dim)

        // self.cross_attn_token_to_image = Attention(
        //     embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        // )
        // self.norm2 = nn.LayerNorm(embedding_dim)

        // self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        // self.norm3 = nn.LayerNorm(embedding_dim)

        // self.norm4 = nn.LayerNorm(embedding_dim)
        // self.cross_attn_image_to_token = Attention(
        //     embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        // )

        // self.skip_first_layer_pe = skip_first_layer_pe
    }

    pub fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        query_pe: &Tensor,
        key_pe: &Tensor,
    ) -> (Tensor, Tensor) {
        unimplemented!()
        // # Self attention block
        // if self.skip_first_layer_pe:
        //     queries = self.self_attn(q=queries, k=queries, v=queries)
        // else:
        //     q = queries + query_pe
        //     attn_out = self.self_attn(q=q, k=q, v=queries)
        //     queries = queries + attn_out
        // queries = self.norm1(queries)

        // # Cross attention block, tokens attending to image embedding
        // q = queries + query_pe
        // k = keys + key_pe
        // attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        // queries = queries + attn_out
        // queries = self.norm2(queries)

        // # MLP block
        // mlp_out = self.mlp(queries)
        // queries = queries + mlp_out
        // queries = self.norm3(queries)

        // # Cross attention block, image embedding attending to tokens
        // q = queries + query_pe
        // k = keys + key_pe
        // attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        // keys = keys + attn_out
        // keys = self.norm4(keys)

        // return queries, keys
    }
}

// An attention layer that allows for downscaling the size of the embedding
//     after projection to queries, keys, and values.
pub struct Attention {}

impl Attention {
    pub fn new(embedding_dim: i64, num_heads: i64, downsample_rate: Option<i64>) -> Self {
        let downsample_rate = downsample_rate.unwrap_or(1);
        unimplemented!()
        // super().__init__()
        // self.embedding_dim = embedding_dim
        // self.internal_dim = embedding_dim // downsample_rate
        // self.num_heads = num_heads
        // assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        // self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        // self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        // self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        // self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
    }

    fn _separate_heads(&self, x: Tensor, num_heads: i64) -> Tensor {
        unimplemented!()
        // b, n, c = x.shape
        // x = x.reshape(b, n, num_heads, c // num_heads)
        // return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head
    }

    fn _recombine_heads(&self, x: Tensor) -> Tensor {
        unimplemented!()
        // b, n_heads, n_tokens, c_per_head = x.shape
        // x = x.transpose(1, 2)
        // return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        unimplemented!()
        // # Input projections
        // q = self.q_proj(q)
        // k = self.k_proj(k)
        // v = self.v_proj(v)

        // # Separate into heads
        // q = self._separate_heads(q, self.num_heads)
        // k = self._separate_heads(k, self.num_heads)
        // v = self._separate_heads(v, self.num_heads)

        // # Attention
        // _, _, _, c_per_head = q.shape
        // attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        // attn = attn / math.sqrt(c_per_head)
        // attn = torch.softmax(attn, dim=-1)

        // # Get output
        // out = attn @ v
        // out = self._recombine_heads(out)
        // out = self.out_proj(out)

        // return out
    }
}
