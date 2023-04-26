mod mlp;
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};
use mlp::MLP;

use crate::burn_helpers::{ConvTranspose2d, TensorHelpers};

use super::{
    common::{activation::Activation, layer_norm_2d::LayerNorm2d},
    transformer::TwoWayTransformer,
};

#[derive(Debug, Module)]
pub struct MaskDecoder<B: Backend> {
    _transformer_dim: usize,
    transformer: TwoWayTransformer<B>,
    _num_multimask_outputs: usize,
    iou_token: Embedding<B>,
    pub num_mask_tokens: usize,
    mask_tokens: Embedding<B>,
    output_hypernetworks_mlps: Vec<MLP<B>>,
    iou_prediction_head: MLP<B>,
    seq1: ConvTranspose2d<B>,
    seq2: LayerNorm2d<B>,
    seq3: Activation,
    seq4: ConvTranspose2d<B>,
    seq5: Activation,
}
impl<B: Backend> MaskDecoder<B> {
    pub fn new(
        transformer_dim: usize,
        transformer: TwoWayTransformer<B>,
        num_multimask_outputs: usize,
        activation: Activation,
        iou_head_depth: usize,
        iou_head_hidden_dim: usize,
    ) -> Self {
        let iou_token = EmbeddingConfig::new(1, transformer_dim).init();
        let num_mask_tokens = num_multimask_outputs + 1;

        let mask_tokens = EmbeddingConfig::new(num_mask_tokens, transformer_dim).init();

        let seq1 = ConvTranspose2d::new(transformer_dim, transformer_dim / 4, 2, 2);
        let seq2 = LayerNorm2d::new(transformer_dim / 4, None);
        let seq3 = activation;
        let seq4 = ConvTranspose2d::new(transformer_dim / 4, transformer_dim / 8, 2, 2);
        let seq5 = activation;

        let mut output_hypernetworks_mlps = Vec::new();
        for _ in 0..num_mask_tokens {
            output_hypernetworks_mlps.push(MLP::new(
                transformer_dim,
                transformer_dim,
                transformer_dim / 8,
                3,
                false,
            ));
        }
        let iou_prediction_head = MLP::new(
            transformer_dim,
            iou_head_hidden_dim,
            num_mask_tokens,
            iou_head_depth,
            false,
        );
        MaskDecoder {
            _transformer_dim: transformer_dim,
            transformer,
            _num_multimask_outputs: num_multimask_outputs,
            num_mask_tokens,
            iou_token,
            mask_tokens,
            output_hypernetworks_mlps,
            iou_prediction_head,
            seq1,
            seq2,
            seq3,
            seq4,
            seq5,
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
        image_embeddings: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        sparse_prompt_embeddings: Tensor<B, 3>,
        dense_prompt_embeddings: Tensor<B, 4>,
        multimask_output: bool,
    ) -> (Tensor<B, 4>, Tensor<B, 2>) {
        let (masks, iou_pred) = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        );

        if multimask_output {
            (
                masks.narrow(1, 1, masks.shape().dims[1] - 1),
                iou_pred.narrow(1, 1, iou_pred.shape().dims[1] - 1),
            )
        } else {
            (masks.narrow(1, 0, 1), iou_pred.narrow(1, 0, 1))
        }
    }

    /// Predicts masks. See 'forward' for more details.
    pub fn predict_masks(
        &self,
        image_embeddings: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        sparse_prompt_embeddings: Tensor<B, 3>,
        dense_prompt_embeddings: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 2>) {
        let ws1 = self.iou_token.clone().into_record().weight.val();
        let ws2 = self.mask_tokens.clone().into_record().weight.val();
        let output_tokens = Tensor::cat(vec![ws1, ws2], 0);
        let output_tokens = output_tokens.unsqueeze().expand(
            vec![
                sparse_prompt_embeddings.shape().dims[0],
                usize::MAX,
                usize::MAX,
            ],
            false,
        );
        let tokens = Tensor::cat(vec![output_tokens, sparse_prompt_embeddings], 1);

        let src = image_embeddings//.repeat_interleave_self_int(tokens.shape().dims[0], 0, None)
            + dense_prompt_embeddings;
        let pos_src = image_pe; //.repeat_interleave_self_int(tokens.shape().dims[0], 0, None);

        let shape = src.shape().dims;
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (hs, src) = self.transformer.forward(src, pos_src, tokens);
        let iou_token_out = hs.narrow(1, 0, 1);
        let mask_tokens_out = hs.narrow(1, 1, self.num_mask_tokens);

        let src = src.transpose().reshape([b, c, h, w]);
        // let src = src.transpose(1, 2).view([b, c, h, w]); //Todo it was this one
        let upscaled_embedding = self.output_upscaling(src);
        let mut hyper_in_list: Vec<Tensor<B, 3>> = vec![];
        for i in 0..self.num_mask_tokens {
            let input = mask_tokens_out.narrow(1, i, 1); //.squeeze_dim(1);
            let item = self.output_hypernetworks_mlps[i as usize].forward(input);
            hyper_in_list.push(item);
        }
        let hyper_in = Tensor::stack(hyper_in_list, 1);

        let shape = upscaled_embedding.shape().dims;
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let masks = hyper_in
            .matmul(upscaled_embedding.reshape([b, c, h * w]))
            .reshape_max([b, usize::MAX, h, w]);

        let iou_pred = self.iou_prediction_head.forward(iou_token_out).unsqueeze();
        return (masks, iou_pred);
    }

    fn output_upscaling<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = x;
        x = self.seq1.forward(x);
        x = self.seq2.forward(x);
        x = self.seq3.forward(x);
        x = self.seq4.forward(x);
        x = self.seq5.forward(x);
        x
    }
}

#[cfg(test)]
mod test {

    use crate::{
        modeling::{common::activation::Activation, transformer::TwoWayTransformer},
        tests::helpers::{random_tensor, Test, TestBackend},
    };

    #[test]
    fn test_mask_decoder() {
        let gelu = Activation::GELU;
        let relu = Activation::ReLU;
        let two_way_transformer = TwoWayTransformer::new(2, 64, 2, 512, Some(relu), Some(2));
        let mut mask_decoder =
            super::MaskDecoder::<TestBackend>::new(64, two_way_transformer, 3, gelu, 3, 64);
        let file = Test::open("mask_decoder_predict");
        mask_decoder = file.load(mask_decoder);
        
        // Forward
        let image_embedding = random_tensor([1, 64, 16, 16], 1);
        let image_pe = random_tensor([1, 64, 16, 16], 2);
        let sparse_prompt_embeddings = random_tensor([16, 2, 64], 3);
        let dense_prompt_embeddings = random_tensor([16, 64, 16, 16], 4);
        let (masks, iou_pred) = mask_decoder.forward(
            image_embedding.clone(),
            image_pe.clone(),
            sparse_prompt_embeddings.clone(),
            dense_prompt_embeddings.clone(),
            true,
        );
        file.compare("image_embedding", image_embedding);
        file.compare("image_pe", image_pe);
        file.compare("sparse_prompt_embeddings", sparse_prompt_embeddings);
        file.compare("dense_prompt_embeddings", dense_prompt_embeddings);
        file.compare("masks", masks);
        file.compare("iou_pred", iou_pred);
    }

    #[test]
    fn test_mask_decoder_predict() {
        let gelu = Activation::GELU;
        let relu = Activation::ReLU;
        let two_way_transformer = TwoWayTransformer::new(2, 64, 2, 512, Some(relu), Some(2));
        let mut mask_decoder =
            super::MaskDecoder::<TestBackend>::new(64, two_way_transformer, 3, gelu, 3, 64);
        let file = Test::open("mask_decoder");
        mask_decoder = file.load(mask_decoder);

        // Forward
        let image_embedding = random_tensor([1, 64, 16, 16], 1);
        let image_pe = random_tensor([1, 64, 16, 16], 2);
        let sparse_prompt_embeddings = random_tensor([16, 2, 64], 3);
        let dense_prompt_embeddings = random_tensor([16, 64, 16, 16], 4);
        let (masks, iou_pred) = mask_decoder.predict_masks(
            image_embedding.clone(),
            image_pe.clone(),
            sparse_prompt_embeddings.clone(),
            dense_prompt_embeddings.clone(),
        );
        file.compare("image_embedding", image_embedding);
        file.compare("image_pe", image_pe);
        file.compare("sparse_prompt_embeddings", sparse_prompt_embeddings);
        file.compare("dense_prompt_embeddings", dense_prompt_embeddings);
        file.compare("masks", masks);
        file.compare("iou_pred", iou_pred);
    }
}
