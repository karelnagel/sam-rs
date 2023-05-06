use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::{Bool, Float, Int};

use crate::burn_helpers::{TensorHelpers, TensorSlice, ToFloat};
use crate::{
    modeling::{
        image_encoder::ImageEncoderViT, mask_decoder::MaskDecoder, prompt_encoder::PromptEncoder,
    },
    sam_predictor::{ImageFormat, Size},
};

#[derive(Debug, Module)]
pub struct Sam<B: Backend> {
    pub image_encoder: ImageEncoderViT<B>,
    pub prompt_encoder: PromptEncoder<B>,
    pub mask_decoder: MaskDecoder<B>,
    pub pixel_mean: [f32; 3],
    pub pixel_std: [f32; 3],
    pub mask_threshold: f64,
    pub image_format: ImageFormat,
}
#[derive(Debug)]
pub struct Input<B: Backend> {
    pub image: Tensor<B, 3, Int>,
    pub original_size: Size,
    pub boxes: Option<Tensor<B, 2>>,
    pub points: Option<(Tensor<B, 3>, Tensor<B, 2>)>,
    pub mask_inputs: Option<Tensor<B, 4>>,
}
pub struct Output<B: Backend> {
    pub masks: Tensor<B, 4, Bool>,
    pub iou_predictions: Tensor<B, 2, Float>,
    pub low_res_logits: Option<Tensor<B, 4, Float>>,
}
impl<B: Backend> Sam<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    /// # SAM predicts object masks from an image and input prompts.
    ///
    /// Arguments:
    ///   - image_encoder (ImageEncoderViT): The backbone used to encode the
    ///     image into image embeddings that allow for efficient mask prediction.
    ///   - prompt_encoder (PromptEncoder): Encodes various types of input prompts.
    ///   - mask_decoder (MaskDecoder): Predicts masks from the image embeddings
    ///     and encoded prompts.
    ///   - pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
    ///   - pixel_std (list(float)): Std values for normalizing pixels in the input image.
    pub fn new(
        image_encoder: ImageEncoderViT<B>,
        prompt_encoder: PromptEncoder<B>,
        mask_decoder: MaskDecoder<B>,
        pixel_mean: Option<[f32; 3]>,
        pixel_std: Option<[f32; 3]>,
    ) -> Self {
        let pixel_mean = pixel_mean.unwrap_or([123.675, 116.28, 103.53]);
        let pixel_std = pixel_std.unwrap_or([58.395, 57.12, 57.375]);

        Self {
            image_encoder,
            prompt_encoder,
            mask_decoder,
            pixel_mean,
            pixel_std,
            mask_threshold: 0.0,
            image_format: ImageFormat::RGB,
        }
    }
    fn pixel_mean(&self) -> Tensor<B, 3> {
        Tensor::of_slice(self.pixel_mean.to_vec(), [self.pixel_mean.len()]).reshape_max([
            usize::MAX,
            1,
            1,
        ])
    }
    fn pixel_std(&self) -> Tensor<B, 3> {
        Tensor::of_slice(self.pixel_std.to_vec(), [self.pixel_std.len()]).reshape_max([
            usize::MAX,
            1,
            1,
        ])
    }

    /// Predicts masks end-to-end from provided images and prompts.
    /// If prompts are not known in advance, using SamPredictor is
    /// recommended over calling the model directly.
    ///
    /// Arguments:
    ///   - batched_input (list(dict)): A list over input images, each a
    ///     dictionary with the following keys. A prompt key can be
    ///     excluded if it is not present.
    ///       - `image`: The image as a torch tensor in 3xHxW format,
    ///         already transformed for input to the model.
    ///       - `original_size`: (tuple(int, int)) The original size of
    ///         the image before transformation, as (H, W).
    ///       - `point_coords`: (torch.Tensor) Batched point prompts for
    ///         this image, with shape BxNx2. Already transformed to the
    ///         input frame of the model.
    ///       - `point_labels`: (torch.Tensor) Batched labels for point prompts,
    ///         with shape BxN.
    ///       - `boxes`: (torch.Tensor) Batched box inputs, with shape Bx4.
    ///         Already transformed to the input frame of the model.
    ///       - `mask_inputs`: (torch.Tensor) Batched mask inputs to the model,
    ///         in the form Bx1xHxW.
    ///   - multimask_output (bool): Whether the model should predict multiple
    ///     disambiguating masks, or return a single mask.
    ///
    /// Returns:
    ///   (list(dict)): A list over input images, where each element is
    ///     as dictionary with the following keys.
    ///       - `masks`: (torch.Tensor) Batched binary mask predictions,
    ///         with shape BxCxHxW, where B is the number of input prompts,
    ///         C is determined by multimask_output, and (H, W) is the
    ///         original size of the image.
    ///       - `iou_predictions`: (torch.Tensor) The model's predictions
    ///         of mask quality, in shape BxC.
    ///       - `low_res_logits`: (torch.Tensor) Low resolution logits with
    ///         shape BxCxHxW, where H=W=256. Can be passed as mask input
    ///         to subsequent iterations of prediction.
    pub fn forward(
        &mut self,
        batched_input: Vec<Input<B>>,
        multimask_output: bool,
    ) -> Vec<Output<B>> {
        let processed_images = batched_input
            .iter()
            .map(|x| self.preprocess(x.image.clone()))
            .collect::<Vec<_>>();
        let input_images = Tensor::stack(processed_images.clone(), 0);
        let image_embeddings = self.image_encoder.forward(input_images.clone());
        let image_embeddings_vec: Vec<Tensor<B, 3>> = image_embeddings.clone().unbind(0);
        assert_eq!(image_embeddings_vec.len(), batched_input.len());
        let mut outputs: Vec<Output<B>> = vec![];
        for i in 0..batched_input.len() {
            let image_record = batched_input.get(i).unwrap();
            let curr_embedding: Tensor<B, 3> = image_embeddings_vec.clone()[i].clone();
            let (sparse_embeddings, dense_embeddings) = self.prompt_encoder.forward(
                image_record.points.clone(),
                image_record.boxes.clone(),
                image_record.mask_inputs.clone(),
            );
            let image_pe = self.prompt_encoder.get_dense_pe();
            let (low_res_masks, iou_predictions) = self.mask_decoder.forward(
                curr_embedding.clone().unsqueeze(),
                image_pe,
                sparse_embeddings,
                dense_embeddings,
                multimask_output,
            );
            let size = image_record.image.dims();
            let masks = self.postprocess_masks(
                low_res_masks.clone(),
                Size(size[size.len() - 2], size[size.len() - 1]),
                image_record.original_size,
            );
            let masks = masks.clone().greater_elem(self.mask_threshold);
            outputs.push(Output {
                masks,
                iou_predictions,
                low_res_logits: Some(low_res_masks),
            })
        }
        outputs
    }

    /// Remove padding and upscale masks to the original image size.
    /// Arguments:
    ///   masks (torch.Tensor): Batched masks from the mask_decoder,
    ///     in BxCxHxW format.
    ///   input_size (tuple(int, int)): The size of the image input to the
    ///     model, in (H, W) format. Used to remove padding.
    ///   original_size (tuple(int, int)): The original size of the image
    ///     before resizing for input to the model, in (H, W) format.
    /// Returns:
    ///   (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
    ///     is given by original_size.
    pub fn postprocess_masks(
        &self,
        masks: Tensor<B, 4, Float>,
        input: Size,
        original: Size,
    ) -> Tensor<B, 4, Float> {
        let output_size = vec![self.image_encoder.img_size, self.image_encoder.img_size];
        let masks = masks.upsample_bilinear2d::<4>(output_size, false, None, None);
        let masks: Tensor<B, 4> = masks.narrow(2, 0, input.0);
        let masks = masks.narrow(3, 0, input.1);
        let masks = masks.upsample_bilinear2d(vec![original.0, original.1], false, None, None);
        masks
    }

    /// Normalize pixel values and pad to a square input.
    pub fn preprocess<const D: usize>(&self, x: Tensor<B, D, Int>) -> Tensor<B, D, Float> {
        let x: Tensor<B, D, Float> =
            (x.to_float() - self.pixel_mean().unsqueeze()) / self.pixel_std().unsqueeze();
        let size = x.dims();
        let (h, w) = (size[D - 2], size[D - 1]);

        let padh = self.image_encoder.img_size - h;
        let padw = self.image_encoder.img_size - w;
        let x = x.pad(&[0, padw, 0, padh], "constant", 0.);
        x
    }
}

#[cfg(test)]
mod test {
    use crate::{
        build_sam::build_sam_test,
        sam_predictor::Size,
        tests::helpers::{
            load_module, random_tensor, random_tensor_int, Test, TestBackend, TEST_CHECKPOINT,
        },
    };

    use super::Input;

    #[test]
    fn test_sam_forward_boxes() {
        let mut sam = build_sam_test::<TestBackend>(None);
        sam = load_module("sam_forward_boxes", sam); // Todo switch back to checkpoint
        let input = vec![
            Input {
                image: random_tensor_int([3, 8, 8], 1, 255.),
                boxes: Some(random_tensor([4, 4], 1)),
                original_size: Size(100, 200),
                mask_inputs: None,
                points: None,
            },
            Input {
                image: random_tensor_int([3, 8, 8], 1, 255.),
                boxes: Some(random_tensor([4, 4], 1)),
                original_size: Size(50, 80),
                mask_inputs: None,
                points: None,
            },
        ];
        let output = sam.forward(input, false);
        let file = Test::open("sam_forward_boxes");
        for (i, out) in output.iter().enumerate() {
            file.compare(format!("masks{}", i).as_str(), out.masks.clone());
            // file.compare(
            //     format!("iou_predictions{}", i).as_str(),
            //     out.iou_predictions.clone(),
            // );
            // if let Some(low_res_logits) = out.low_res_logits.clone() {
            //     file.compare(format!("low_res_logits{}", i).as_str(), low_res_logits);
            // } //Todo
        }
    }
    #[test]
    fn test_sam_forward_points() {
        let mut sam = build_sam_test::<TestBackend>(None);
        sam = load_module("sam_forward_points", sam); // Todo switch back to checkpoint
        let input = vec![
            Input {
                image: random_tensor_int([3, 8, 8], 1, 255.),
                boxes: None,
                original_size: Size(100, 200),
                mask_inputs: None,
                points: Some((random_tensor([4, 2, 2], 1), random_tensor([4, 2], 1))),
            },
            Input {
                image: random_tensor_int([3, 8, 8], 2, 255.),
                boxes: None,
                original_size: Size(50, 80),
                mask_inputs: None,
                points: Some((random_tensor([4, 2, 2], 2), random_tensor([4, 2], 2))),
            },
        ];
        let output = sam.forward(input, false);
        let file = Test::open("sam_forward_points");
        for (i, out) in output.iter().enumerate() {
            file.compare(format!("masks{}", i).as_str(), out.masks.clone());
            // file.compare(
            //     format!("iou_predictions{}", i).as_str(),
            //     out.iou_predictions.clone(),
            // );
            // if let Some(low_res_logits) = out.low_res_logits.clone() {
            //     file.compare(format!("low_res_logits{}", i).as_str(), low_res_logits);
            // } //Todo
        }
    }
    #[test]
    fn test_sam_postprocess_masks() {
        let sam = build_sam_test::<TestBackend>(Some(TEST_CHECKPOINT));

        let masks = random_tensor([4, 1, 256, 256], 1);
        let input = Size(684, 1024);
        let original = Size(534, 800);
        let output = sam.postprocess_masks(masks.clone(), input, original);
        let file = Test::open("sam_postprocess_masks");
        file.compare("input_size", input);
        file.compare("masks", masks);
        file.compare("output", output);
    }
    #[test]
    fn test_sam_preprocess() {
        let sam = build_sam_test::<TestBackend>(None);
        // sam = load_module("sam_preprocess", sam);

        let input = random_tensor_int([3, 171, 128], 1, 255.);
        let output = sam.preprocess(input.clone());
        let file = Test::open("sam_preprocess");
        file.compare("input", input);
        file.compare("output", output);
    }
}
