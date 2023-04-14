use tch::Tensor;

use crate::{
    modeling::{
        image_encoder::ImageEncoderViT, mask_decoder::MaskDecoder, prompt_encoder::PromptEncoder,
    },
    sam_predictor::{ImageFormat, Size},
};

#[derive(Debug)]
pub struct Sam {
    pub image_encoder: ImageEncoderViT,
    pub prompt_encoder: PromptEncoder,
    pub mask_decoder: MaskDecoder,
    pub pixel_mean: Tensor,
    pub pixel_std: Tensor,
    pub mask_threshold: f32,
    pub image_format: ImageFormat,
}
pub struct Input {
    pub image: Tensor,
    pub original_size: Size,
    pub point_coords: Tensor,
    pub point_labels: Tensor,
    pub boxes: Tensor,
    pub mask_inputs: Tensor,
}
pub struct Output {
    pub masks: Tensor,
    pub iou_predictions: Tensor,
    pub low_res_masks: Tensor,
}
impl Sam {
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
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: Option<&[f32]>,
        pixel_std: Option<&[f32]>,
    ) -> Self {
        let pixel_mean = Tensor::of_slice(pixel_mean.unwrap_or([123.675, 116.28, 103.53].as_ref()))
            .view([-1, 1, 1]);
        let pixel_std = Tensor::of_slice(pixel_std.unwrap_or([58.395, 57.12, 57.375].as_ref()))
            .view([-1, 1, 1]);

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
    pub fn forward(&self, batched_input: Vec<Input>, multimask_output: bool) -> Vec<Output> {
        let input_images = Tensor::stack(
            &batched_input
                .iter()
                .map(|x| self.preprocess(x.image.copy()))
                .collect::<Vec<_>>(),
            0,
        );
        let image_embeddings = self.image_encoder.forward(&input_images);

        let mut outputs: Vec<Output> = vec![];
        for i in 0..batched_input.len() {
            let image_record = batched_input.get(i).unwrap();
            let curr_embedding = image_embeddings.get(i as i64);

            let points = Some((
                image_record.point_coords.copy(),
                image_record.point_labels.copy(),
            ));
            let (sparse_embeddings, dense_embeddings) = self.prompt_encoder.forward(
                points,
                Some(image_record.boxes.copy()),
                Some(image_record.mask_inputs.copy()),
            );
            let (low_res_masks, iou_predictions) = self.mask_decoder.forward(
                &curr_embedding.unsqueeze(0),
                self.prompt_encoder.get_dense_pe(),
                sparse_embeddings,
                dense_embeddings,
                multimask_output,
            );
            let size = image_record.image.size();
            let masks = self.postprocess_masks(
                &low_res_masks,
                &Size(size[1], size[2]),
                &image_record.original_size,
            );
            // let masks = masks > self.mask_threshold; // Todo
            outputs.push(Output {
                masks,
                iou_predictions,
                low_res_masks,
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
    pub fn postprocess_masks(&self, masks: &Tensor, input: &Size, original: &Size) -> Tensor {
        let output_size: &[i64; 2] = &[self.image_encoder.img_size, self.image_encoder.img_size];
        let masks = masks.upsample_bilinear2d(output_size, false, None, None);
        let masks = masks.slice(2, 0, input.0, 1);
        let masks = masks.slice(3, 0, input.1, 1);
        let output_size: &[i64; 2] = &[original.0, original.1];
        let masks = masks.upsample_bilinear2d(output_size, false, None, None);
        masks
    }

    /// Normalize pixel values and pad to a square input.
    pub fn preprocess(&self, x: Tensor) -> Tensor {
        let x = (&x - &self.pixel_mean) / &self.pixel_std;
        let (h, w) = x.size2().unwrap();

        let padh = self.image_encoder.img_size - h;
        let padw = self.image_encoder.img_size - w;
        let x = x.constant_pad_nd(&[0, padw, 0, padh]);
        x
    }
}
