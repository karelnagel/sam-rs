mod mlp;
use mlp::MLP;
use tch::{
    nn::{self, Embedding, Module, Sequential},
    Tensor,
};

use super::{
    common::{activation::Activation, layer_norm_2d::LayerNorm2d},
    transformer::TwoWayTransformer,
};

#[derive(Debug)]
pub struct MaskDecoder {
    _transformer_dim: i64,
    transformer: TwoWayTransformer,
    _num_multimask_outputs: i64,
    iou_token: Embedding,
    pub num_mask_tokens: i64,
    mask_tokens: Embedding,
    output_upscaling: Sequential,
    output_hypernetworks_mlps: Vec<MLP>,
    iou_prediction_head: MLP,
}
impl Module for MaskDecoder{
    fn forward(&self, _: &Tensor) -> Tensor {
       unimplemented!()
    }
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

        let output_upscaling = nn::seq()
            .add(nn::conv_transpose2d(
                vs,
                transformer_dim,
                transformer_dim / 4,
                2,
                nn::ConvTransposeConfig {
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add(LayerNorm2d::new(vs, transformer_dim / 4, None))
            .add(activation)
            .add(nn::conv_transpose2d(
                vs,
                transformer_dim / 4,
                transformer_dim / 8,
                2,
                nn::ConvTransposeConfig {
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add(activation);

        let mut output_hypernetworks_mlps = Vec::new();
        for _ in 0..num_mask_tokens {
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
            _transformer_dim: transformer_dim,
            transformer,
            _num_multimask_outputs: num_multimask_outputs,
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
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
    ) -> (Tensor, Tensor) {
        let (masks, iou_pred) = self.predict_masks(
            image_embeddings,
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
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
    ) -> (Tensor, Tensor) {
        let output_tokens = Tensor::cat(&[&self.iou_token.ws, &self.mask_tokens.ws], 0);
        let output_tokens = output_tokens
            .unsqueeze(0)
            .expand(&[sparse_prompt_embeddings.size()[0], -1, -1], false);
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1);

        let src = image_embeddings.repeat_interleave_self_int(tokens.size()[0], 0, None)
            + dense_prompt_embeddings;
        let pos_src = image_pe.repeat_interleave_self_int(tokens.size()[0], 0, None);

        let (b, c, h, w) = src.size4().unwrap();
        let (hs, src) = self.transformer.forward(&src, &pos_src, &tokens);
        let iou_token_out = hs.narrow(1, 0, 1).squeeze_dim(1);
        let mask_tokens_out = hs.narrow(1, 1, self.num_mask_tokens);

        let src = src.transpose(1, 2).view([b, c, h, w]);
        let upscaled_embedding = self.output_upscaling.forward(&src);
        let mut hyper_in_list: Vec<Tensor> = vec![];
        for i in 0..self.num_mask_tokens {
            let input = &mask_tokens_out.narrow(1, i, 1).squeeze_dim(1);
            let item = self.output_hypernetworks_mlps[i as usize].forward(input);
            hyper_in_list.push(item);
        }
        let hyper_in = Tensor::stack(&hyper_in_list, 1);

        let (b, c, h, w) = upscaled_embedding.size4().unwrap();

        let masks = hyper_in
            .matmul(&upscaled_embedding.view([b, c, h * w]))
            .view([b, -1, h, w]);

        let iou_pred = self.iou_prediction_head.forward(&iou_token_out);
        return (masks, iou_pred);
    }
}

#[cfg(test)]
mod test {
    use tch::{nn, Device};

    use crate::{
        modeling::{
            common::{
                activation::{Activation, ActivationType},
                layer_norm_2d::LayerNorm2d,
            },
            transformer::TwoWayTransformer,
        },
        tests::{
            helpers::{random_tensor, TestFile},
            mocks::Mock,
        },
    };

    impl Mock for super::MaskDecoder {
        fn mock(&mut self) {
            self.transformer.mock();
            self.iou_token.mock();
            self.mask_tokens.mock();
            self.iou_prediction_head.mock();
            for mlp in self.output_hypernetworks_mlps.iter_mut() {
                mlp.mock();
            }
            let vs = nn::VarStore::new(Device::Cpu);
            let mut conv = nn::conv_transpose2d(
                &vs.root(),
                self._transformer_dim,
                self._transformer_dim / 4,
                2,
                nn::ConvTransposeConfig {
                    stride: 2,
                    ..Default::default()
                },
            );
            let layer = LayerNorm2d::new(&vs.root(), self._transformer_dim / 4, None);
            let activation = Activation::new(ActivationType::GELU);
            let mut conv2 = nn::conv_transpose2d(
                &vs.root(),
                self._transformer_dim / 4,
                self._transformer_dim / 8,
                2,
                nn::ConvTransposeConfig {
                    stride: 2,
                    ..Default::default()
                },
            );
            conv.mock();
            conv2.mock();
            self.output_upscaling = nn::seq()
                .add(conv)
                .add(layer)
                .add(activation)
                .add(conv2)
                .add(activation);
        }
    }

    #[test]
    fn test_mask_decoder() {
        let vs = nn::VarStore::new(Device::Cpu);
        let gelu = Activation::new(ActivationType::GELU);
        let relu = Activation::new(ActivationType::ReLU);
        let two_way_transformer =
            TwoWayTransformer::new(&vs.root(), 2, 64, 2, 512, Some(relu), Some(2));
        let mut mask_decoder =
            super::MaskDecoder::new(&vs.root(), 64, two_way_transformer, 3, gelu, 3, 64);
        let file = TestFile::open("mask_decoder");
        file.compare("transformer_dim", mask_decoder._transformer_dim);
        file.compare("num_multimask_outputs", mask_decoder._num_multimask_outputs);
        file.compare("num_mask_tokens", mask_decoder.num_mask_tokens);

        // Mocking
        mask_decoder.mock();

        // Forward
        let image_embedding = random_tensor(&[1, 64, 16, 16], 1);
        let image_pe = random_tensor(&[1, 64, 16, 16], 2);
        let sparse_prompt_embeddings = random_tensor(&[16, 2, 64], 3);
        let dense_prompt_embeddings = random_tensor(&[16, 64, 16, 16], 4);
        let (masks, iou_pred) = mask_decoder.forward(
            &image_embedding,
            &image_pe,
            &sparse_prompt_embeddings,
            &dense_prompt_embeddings,
            true,
        );
        let file = TestFile::open("mask_decoder_forward");
        file.compare("image_embedding", image_embedding);
        file.compare("image_pe", image_pe);
        file.compare("sparse_prompt_embeddings", sparse_prompt_embeddings);
        file.compare("dense_prompt_embeddings", dense_prompt_embeddings);
        file.compare("masks", masks);
        file.compare("iou_pred", iou_pred);
    }

    #[test]
    fn test_mask_decoder_predict() {
        let vs = nn::VarStore::new(Device::Cpu);
        let gelu = Activation::new(ActivationType::GELU);
        let relu = Activation::new(ActivationType::ReLU);
        let two_way_transformer =
            TwoWayTransformer::new(&vs.root(), 2, 64, 2, 512, Some(relu), Some(2));
        let mut mask_decoder =
            super::MaskDecoder::new(&vs.root(), 64, two_way_transformer, 3, gelu, 3, 64);

        // Mocking
        mask_decoder.mock();

        // Forward
        let image_embedding = random_tensor(&[1, 64, 16, 16], 1);
        let image_pe = random_tensor(&[1, 64, 16, 16], 2);
        let sparse_prompt_embeddings = random_tensor(&[16, 2, 64], 3);
        let dense_prompt_embeddings = random_tensor(&[16, 64, 16, 16], 4);
        let (masks, iou_pred) = mask_decoder.predict_masks(
            &image_embedding,
            &image_pe,
            &sparse_prompt_embeddings,
            &dense_prompt_embeddings,
        );
        let file = TestFile::open("mask_decoder_predict");
        file.compare("image_embedding", image_embedding);
        file.compare("image_pe", image_pe);
        file.compare("sparse_prompt_embeddings", sparse_prompt_embeddings);
        file.compare("dense_prompt_embeddings", dense_prompt_embeddings);
        file.compare("masks", masks);
        file.compare("iou_pred", iou_pred);
    }
}
