mod mlp;
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};
use mlp::MLP;

use crate::burn_helpers::TensorHelpers;

use super::{
    common::{
        activation::Activation,
        conv_transpose_2d::{ConvTranspose2d, ConvTranspose2dConfig},
        layer_norm_2d::LayerNorm2d,
    },
    transformer::TwoWayTransformer,
};

#[derive(Debug, Module)]
pub struct MaskDecoder<B: Backend> {
    transformer: TwoWayTransformer<B>,
    iou_token: Embedding<B>,
    pub num_mask_tokens: usize,
    mask_tokens: Embedding<B>,
    output_hypernetworks_mlps: Vec<MLP<B>>,
    iou_prediction_head: MLP<B>,
    output_upscaling0: ConvTranspose2d<B>,
    output_upscaling1: LayerNorm2d<B>,
    output_upscaling2: Activation,
    output_upscaling3: ConvTranspose2d<B>,
    output_upscaling4: Activation,
}
impl<B: Backend> MaskDecoder<B> {
    pub fn new(
        transformer_dim: usize,
        transformer: TwoWayTransformer<B>,
        num_multimask_outputs: Option<usize>,
        activation: Option<Activation>,
        iou_head_depth: Option<usize>,
        iou_head_hidden_dim: Option<usize>,
    ) -> Self {
        let num_multimask_outputs = num_multimask_outputs.unwrap_or(3);
        let activation = activation.unwrap_or(Activation::GELU);
        let iou_head_depth = iou_head_depth.unwrap_or(3);
        let iou_head_hidden_dim = iou_head_hidden_dim.unwrap_or(256);

        let iou_token = EmbeddingConfig::new(1, transformer_dim).init();
        let num_mask_tokens = num_multimask_outputs + 1;

        let mask_tokens = EmbeddingConfig::new(num_mask_tokens, transformer_dim).init();

        let output_upscaling0 =
            ConvTranspose2dConfig::new(transformer_dim, transformer_dim / 4, [2, 2])
                .set_stride([2, 2])
                .init();
        let output_upscaling1 = LayerNorm2d::new(transformer_dim / 4, None);
        let output_upscaling2 = activation;
        let output_upscaling3 =
            ConvTranspose2dConfig::new(transformer_dim / 4, transformer_dim / 8, [2, 2])
                .set_stride([2, 2])
                .init();
        let output_upscaling4 = activation;

        let mut output_hypernetworks_mlps = Vec::new();
        for _ in 0..num_mask_tokens {
            output_hypernetworks_mlps.push(MLP::new(
                transformer_dim,
                transformer_dim,
                transformer_dim / 8,
                3,
                None,
            ));
        }
        let iou_prediction_head = MLP::new(
            transformer_dim,
            iou_head_hidden_dim,
            num_mask_tokens,
            iou_head_depth,
            None,
        );
        MaskDecoder {
            transformer,
            num_mask_tokens,
            iou_token,
            mask_tokens,
            output_hypernetworks_mlps,
            iou_prediction_head,
            output_upscaling0,
            output_upscaling1,
            output_upscaling2,
            output_upscaling3,
            output_upscaling4,
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
                masks.narrow(1, 1, masks.dims()[1] - 1),
                iou_pred.narrow(1, 1, iou_pred.dims()[1] - 1),
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
            vec![sparse_prompt_embeddings.dims()[0], usize::MAX, usize::MAX],
            false,
        );
        let tokens = Tensor::cat(vec![output_tokens, sparse_prompt_embeddings], 1);

        let src = image_embeddings.repeat_interleave_self_int(tokens.dims()[0], Some(0), None)
            + dense_prompt_embeddings;
        let pos_src = image_pe.repeat_interleave_self_int(tokens.dims()[0], Some(0), None);

        let shape = src.dims();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        let (hs, src) = self.transformer.forward(src, pos_src, tokens);
        let iou_token_out = hs.narrow(1, 0, 1);
        let dims = iou_token_out.dims();
        let iou_token_out = iou_token_out.reshape([dims[0], dims[2]]);

        let mask_tokens_out = hs.narrow(1, 1, self.num_mask_tokens);

        let src = src.swap_dims(1, 2).reshape([b, c, h, w]);
        let upscaled_embedding = self.output_upscaling(src);
        let mut hyper_in_list: Vec<Tensor<B, 2>> = vec![];
        for i in 0..self.num_mask_tokens {
            let input = mask_tokens_out.narrow(1, i, 1);
            let dims = input.dims();
            let input = input.reshape([dims[0], dims[2]]);
            let item = self.output_hypernetworks_mlps[i as usize].forward(input);
            hyper_in_list.push(item);
        }
        let hyper_in: Tensor<B, 3> = Tensor::stack(hyper_in_list, 1);

        let shape = upscaled_embedding.dims();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let masks = hyper_in
            .matmul(upscaled_embedding.reshape([b, c, h * w]))
            .reshape_max([b, usize::MAX, h, w]);

        let iou_pred = self.iou_prediction_head.forward(iou_token_out).unsqueeze();
        return (masks, iou_pred);
    }

    fn output_upscaling(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        x = self.output_upscaling0.forward(x);
        x = self.output_upscaling1.forward(x);
        x = self.output_upscaling2.forward(x);
        x = self.output_upscaling3.forward(x);
        x = self.output_upscaling4.forward(x);
        x
    }
}

#[cfg(test)]
mod test {

    use pyo3::{
        types::{PyDict, PyTuple},
        PyResult, Python,
    };

    use crate::{
        modeling::{common::activation::Activation, transformer::TwoWayTransformer},
        python::{
            module_to_file::module_to_file,
            python_data::{random_python_tensor, PythonData},
        },
        tests::helpers::{load_module, TestBackend},
    };

    #[test]
    fn test_mask_decoder_forward() {
        const FILE: &str = "mask_decoder_forward";
        fn python() -> PyResult<(
            PythonData<4>,
            PythonData<4>,
            PythonData<3>,
            PythonData<4>,
            PythonData<4>,
            PythonData<2>,
        )> {
            Python::with_gil(|py| {
                let relu = py.import("torch.nn")?.getattr("ReLU")?;
                let gelu = py.import("torch.nn")?.getattr("GELU")?;
                let transformer = py
                    .import("segment_anything.modeling.transformer")?
                    .getattr("TwoWayTransformer")?;
                let transformer = transformer.call1((2, 64, 2, 512, relu, 2))?;
                let module = py
                    .import("segment_anything.modeling.mask_decoder")?
                    .getattr("MaskDecoder")?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("transformer_dim", 64)?;
                kwargs.set_item("transformer", transformer)?;
                kwargs.set_item("num_multimask_outputs", 3)?;
                kwargs.set_item("activation", gelu)?;
                kwargs.set_item("iou_head_depth", 3)?;
                kwargs.set_item("iou_head_hidden_dim", 256)?;
                let module = module.call((), Some(kwargs))?;
                module_to_file(FILE, py, &module)?;

                let image_embedding = random_python_tensor(py, [1, 64, 16, 16]);
                let image_pe = random_python_tensor(py, [1, 64, 16, 16]);
                let sparse_prompt = random_python_tensor(py, [16, 2, 64]);
                let dense_prompt = random_python_tensor(py, [16, 64, 16, 16]);
                let output =
                    module.call1((image_embedding, image_pe, sparse_prompt, dense_prompt, true))?;
                let output = output.downcast::<PyTuple>()?;
                let masks = output.get_item(0)?;
                let iou_pred = output.get_item(1)?;
                Ok((
                    image_embedding.into(),
                    image_pe.into(),
                    sparse_prompt.into(),
                    dense_prompt.into(),
                    masks.into(),
                    iou_pred.into(),
                ))
            })
        }
        let (image_embedding, image_pe, sparse_prompt, dense_prompt, masks, iou_pred) =
            python().unwrap();
        let two_way_transformer =
            TwoWayTransformer::new(2, 64, 2, 512, Some(Activation::ReLU), Some(2));
        let mut mask_decoder = super::MaskDecoder::<TestBackend>::new(
            64,
            two_way_transformer,
            Some(3),
            Some(Activation::GELU),
            Some(3),
            Some(64),
        );
        mask_decoder = load_module(FILE, mask_decoder);

        // Forward
        let (masks2, iou_pred2) = mask_decoder.forward(
            image_embedding.into(),
            image_pe.into(),
            sparse_prompt.into(),
            dense_prompt.into(),
            true,
        );
        masks.almost_equal(masks2, 5.);
        iou_pred.almost_equal(iou_pred2, None);
    }

    #[test]
    fn test_mask_decoder_predict() {
        const FILE: &str = "mask_decoder_predict";
        fn python() -> PyResult<(
            PythonData<4>,
            PythonData<4>,
            PythonData<3>,
            PythonData<4>,
            PythonData<4>,
            PythonData<2>,
        )> {
            Python::with_gil(|py| {
                let relu = py.import("torch.nn")?.getattr("ReLU")?;
                let gelu = py.import("torch.nn")?.getattr("GELU")?;
                let transformer = py
                    .import("segment_anything.modeling.transformer")?
                    .getattr("TwoWayTransformer")?;
                let transformer = transformer.call1((2, 64, 2, 512, relu, 2))?;
                let module = py
                    .import("segment_anything.modeling.mask_decoder")?
                    .getattr("MaskDecoder")?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("transformer_dim", 64)?;
                kwargs.set_item("transformer", transformer)?;
                kwargs.set_item("num_multimask_outputs", 3)?;
                kwargs.set_item("activation", gelu)?;
                kwargs.set_item("iou_head_depth", 3)?;
                kwargs.set_item("iou_head_hidden_dim", 256)?;
                let module = module.call((), Some(kwargs))?;
                module_to_file(FILE, py, &module)?;

                let image_embedding = random_python_tensor(py, [1, 64, 16, 16]);
                let image_pe = random_python_tensor(py, [1, 64, 16, 16]);
                let sparse_prompt = random_python_tensor(py, [16, 2, 64]);
                let dense_prompt = random_python_tensor(py, [16, 64, 16, 16]);
                let output = module.call_method1(
                    "predict_masks",
                    (image_embedding, image_pe, sparse_prompt, dense_prompt),
                )?;
                let output = output.downcast::<PyTuple>()?;
                let masks = output.get_item(0)?;
                let iou_pred = output.get_item(1)?;
                Ok((
                    image_embedding.into(),
                    image_pe.into(),
                    sparse_prompt.into(),
                    dense_prompt.into(),
                    masks.into(),
                    iou_pred.into(),
                ))
            })
        }
        let (
            image_embedding,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            masks,
            iou_pred,
        ) = python().unwrap();
        let two_way_transformer =
            TwoWayTransformer::new(2, 64, 2, 512, Some(Activation::ReLU), Some(2));
        let mut mask_decoder = super::MaskDecoder::<TestBackend>::new(
            64,
            two_way_transformer,
            Some(3),
            Some(Activation::GELU),
            Some(3),
            Some(64),
        );
        mask_decoder = load_module(FILE, mask_decoder);

        // Forward
        let (masks2, iou_pred2) = mask_decoder.predict_masks(
            image_embedding.into(),
            image_pe.into(),
            sparse_prompt_embeddings.into(),
            dense_prompt_embeddings.into(),
        );
        masks.almost_equal(masks2, 5.);
        iou_pred.almost_equal(iou_pred2, None);
    }
}
