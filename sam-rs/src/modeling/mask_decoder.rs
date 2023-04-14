use tch::{
    nn::{self, Embedding, Module, Sequential},
    Tensor,
};

use super::{common::Activation, transformer::TwoWayTransformer};

#[derive(Debug)]
pub struct MaskDecoder {
    transformer_dim: i64,
    transformer: TwoWayTransformer,
    num_multimask_outputs: i64,
    iou_token: Embedding,
    num_mask_tokens: i64,
    mask_tokens: Embedding,
    output_upscaling: Sequential,
    output_hypernetworks_mlps: Vec<MLP>,
    iou_prediction_head: MLP,
}
impl MaskDecoder {
    pub fn new(
        vs: &nn::Path,
        transformer_dim: i64,
        transformer: TwoWayTransformer,
        num_multimask_outputs: i64,
        activation: Activation,
        iou_head_depth: i64,
        iou_head_hidden_dim: i64,
    ) -> Self {
        let iou_token = nn::embedding(vs, 1, transformer_dim, Default::default());
        let num_mask_tokens = num_multimask_outputs + 1;
        let mask_tokens = nn::embedding(vs, num_mask_tokens, transformer_dim, Default::default());
        // Todo it is wrong
        let output_upscaling = nn::seq()
            .add(nn::conv_transpose2d(
                vs,
                transformer_dim,
                transformer_dim / 4,
                2,
                nn::ConvTransposeConfig {
                    padding: 0,
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add(nn::conv_transpose2d(
                vs,
                transformer_dim / 4,
                transformer_dim / 8,
                2,
                nn::ConvTransposeConfig {
                    padding: 0,
                    stride: 2,
                    ..Default::default()
                },
            ));

        // Todo wrong
        let mut output_hypernetworks_mlps = Vec::new();
        for i in 0..num_mask_tokens {
            output_hypernetworks_mlps.push(MLP::new(
                vs,
                transformer_dim,
                transformer_dim,
                transformer_dim / 8,
                3,
                false,
            ));
        }
        let iou_prediction_head = MLP::new(
            vs,
            transformer_dim,
            iou_head_hidden_dim,
            num_mask_tokens,
            iou_head_depth as usize,
            false,
        );
        MaskDecoder {
            transformer_dim,
            transformer,
            num_multimask_outputs,
            num_mask_tokens,
            iou_token,
            mask_tokens,
            output_upscaling,
            output_hypernetworks_mlps,
            iou_prediction_head,
        }
    }

    // Predict masks given image and prompt embeddings.

    //     Arguments:
    //       image_embeddings (torch.Tensor): the embeddings from the image encoder
    //       image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
    //       sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
    //       dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
    //       multimask_output (bool): Whether to return multiple masks or a single
    //         mask.

    //     Returns:
    //       torch.Tensor: batched predicted masks
    //       torch.Tensor: batched predictions of mask quality
    pub fn forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        multimask_output: bool,
    ) -> (Tensor, Tensor) {
        let (masks, iou_pred) = self.predict_masks(
            image_embeddings.copy(),
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        );
        if multimask_output {
            (
                masks.narrow(1, 1, masks.size()[1] - 1),
                iou_pred.narrow(1, 1, iou_pred.size()[1] - 1),
            )
        } else {
            (masks.narrow(1, 0, 1), iou_pred.narrow(1, 0, 1))
        }
    }

    /// Predicts masks. See 'forward' for more details.
    pub fn predict_masks(
        &self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> (Tensor, Tensor) {
        let output_tokens = Tensor::cat(&[&self.iou_token.ws, &self.mask_tokens.ws], 0);
        let output_tokens = output_tokens
            .unsqueeze(0)
            .expand(&sparse_prompt_embeddings.size(), false);
        let tokens = Tensor::cat(&[output_tokens, sparse_prompt_embeddings], 1);

        let src = Tensor::repeat_interleave(&image_embeddings, tokens.size()[0])
            + dense_prompt_embeddings;
        let pos_src = Tensor::repeat_interleave(&image_pe, tokens.size()[0]);

        let shape = src.size();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        let (hs, src) = self.transformer.forward(&src, pos_src, tokens);
        let iou_token_out = hs.narrow(1, 0, 1);
        let mask_tokens_out = hs.narrow(1, 1, self.num_mask_tokens);

        let src = src.transpose(1, 2).view([b, c, h, w]);
        let upscaled_embedding = self.output_upscaling.forward(&src);
        let hyper_in_list: Vec<Tensor> = (0..self.num_mask_tokens)
            .map(|i| {
                self.output_hypernetworks_mlps[i as usize].forward(&mask_tokens_out.narrow(1, i, 1))
            })
            .collect();
        let hyper_in = Tensor::stack(&hyper_in_list, 1);
        let b = upscaled_embedding.size()[0];
        let c = upscaled_embedding.size()[1];
        let h = upscaled_embedding.size()[2];
        let w = upscaled_embedding.size()[3];

        let masks = hyper_in
            .matmul(&upscaled_embedding.view([b, c, h * w]))
            .view([b, -1, h, w]);

        let iou_pred = self.iou_prediction_head.forward(&iou_token_out);
        return (masks, iou_pred);
    }
}

// 100% GPT-4 code

#[derive(Debug)]
struct MLP {
    layers: nn::Sequential,
    num_layers: usize,
    sigmoid_output: bool,
}

impl MLP {
    fn new(
        vs: &nn::Path,
        input_dim: i64,
        hidden_dim: i64,
        output_dim: i64,
        num_layers: usize,
        sigmoid_output: bool,
    ) -> Self {
        let mut layers = nn::seq();
        let mut last_dim = input_dim;
        for i in 0..num_layers {
            let next_dim = if i == num_layers - 1 {
                output_dim
            } else {
                hidden_dim
            };
            layers = layers.add(nn::linear(
                vs / format!("layer_{}", i),
                last_dim,
                next_dim,
                Default::default(),
            ));
            last_dim = next_dim;
        }
        Self {
            layers,
            num_layers,
            sigmoid_output,
        }
    }
}
impl nn::Module for MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        let intermediate_outputs = self.layers.forward_all(x, None);
        let mut x = intermediate_outputs[0].shallow_clone();
        for i in 1..self.num_layers {
            x = if i < self.num_layers - 1 {
                x.relu()
            } else {
                intermediate_outputs[i].shallow_clone()
            };
        }
        if self.sigmoid_output {
            x.sigmoid()
        } else {
            x
        }
    }
}
