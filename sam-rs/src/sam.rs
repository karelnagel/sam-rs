use tch::{nn::Module, Tensor};

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
    pub mask_threshold: f64,
    pub image_format: ImageFormat,
}
#[derive(Debug)]
pub struct Input<'a> {
    pub image: Tensor,
    pub boxes: Tensor,
    pub original_size: Size,
    pub point_coords: Option<(&'a Tensor, &'a Tensor)>,
    pub mask_inputs: Option<&'a Tensor>,
}
pub struct Output {
    pub masks: Tensor,
    pub iou_predictions: Tensor,
    pub low_res_logits: Option<Tensor>,
}
impl Module for Sam {
    fn forward(&self, _: &Tensor) -> Tensor {
        unimplemented!()
    }
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
                .map(|x| self.preprocess(&x.image))
                .collect::<Vec<_>>(),
            0,
        );
        let image_embeddings = self.image_encoder.forward(&input_images);
        let mut outputs: Vec<Output> = vec![];
        for i in 0..batched_input.len() {
            let image_record = batched_input.get(i).unwrap();
            let curr_embedding = image_embeddings.get(i as i64);
            let (sparse_embeddings, dense_embeddings) = self.prompt_encoder.forward(
                image_record.point_coords,
                Some(&image_record.boxes),
                image_record.mask_inputs,
            );
            let image_embeddings = curr_embedding.unsqueeze(0);
            let image_pe = self.prompt_encoder.get_dense_pe();
            let (low_res_masks, iou_predictions) = self.mask_decoder.forward(
                &image_embeddings,
                &image_pe,
                &sparse_embeddings,
                &dense_embeddings,
                multimask_output,
            );
            let size = image_record.image.size();
            let masks = self.postprocess_masks(
                &low_res_masks,
                &Size(size[size.len() - 2], size[size.len() - 1]),
                &image_record.original_size,
            );
            let masks = masks.gt(self.mask_threshold);
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
    pub fn preprocess(&self, x: &Tensor) -> Tensor {
        let x = (x - &self.pixel_mean) / &self.pixel_std;
        let size = x.size();
        let h = size[size.len() - 2];
        let w = size[size.len() - 1];

        let padh = self.image_encoder.img_size - h;
        let padw = self.image_encoder.img_size - w;
        let x = x.constant_pad_nd(&[0, padw, 0, padh]);
        x
    }
}

#[cfg(test)]
mod test {
    use crate::{
        build_sam::build_sam_vit_b,
        sam_predictor::Size,
        tests::{
            helpers::{random_tensor, TestFile},
            mocks::Mock,
        },
    };

    use super::{Input, Sam};

    impl Mock for Sam {
        fn mock(&mut self) {
            self.prompt_encoder.mock();
            self.mask_decoder.mock();
            self.image_encoder.mock();
        }
    }

    #[test]
    fn test_sam_forward() {
        let mut sam = build_sam_vit_b(None);
        sam.mock();
        let input = vec![
            Input {
                image: random_tensor(&[3, 171, 128], 1),
                boxes: random_tensor(&[4, 4], 1),
                original_size: Size(300, 450),
                mask_inputs: None,
                point_coords: None,
            },
            Input {
                image: random_tensor(&[3, 171, 128], 1),
                boxes: random_tensor(&[4, 4], 1),
                original_size: Size(133, 200),
                mask_inputs: None,
                point_coords: None,
            },
        ];
        let output = sam.forward(input, false);
        let file = TestFile::open("sam_forward");
        for (i, out) in output.iter().enumerate() {
            file.compare(format!("masks{}", i).as_str(), out.masks.copy());
            file.compare(
                format!("iou_predictions{}", i).as_str(),
                out.iou_predictions.copy(),
            );
            if let Some(low_res_logits) = &out.low_res_logits {
                file.compare(
                    format!("low_res_logits{}", i).as_str(),
                    low_res_logits.copy(),
                );
            }
        }
    }
    #[test]
    fn test_sam_postprocess_masks() {
        let mut sam = build_sam_vit_b(None);
        sam.mock();

        let masks = random_tensor(&[4, 1, 256, 256], 1);
        let input = Size(684, 1024);
        let original = Size(534, 800);
        let output = sam.postprocess_masks(&masks, &input, &original);
        let file = TestFile::open("sam_postprocess_masks");
        file.compare("input_size", input);
        file.compare("masks", masks);
        file.compare("output", output);
    }
    #[test]
    fn test_sam_preprocess() {
        let mut sam = build_sam_vit_b(None);
        sam.mock();

        let input = random_tensor(&[1, 3, 171, 128], 1);
        let output = sam.preprocess(&input);
        let file = TestFile::open("sam_preprocess");
        file.compare("input", input);
        file.compare("output", output);
    }
}
