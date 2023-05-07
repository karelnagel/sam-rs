use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::burn_helpers::ToFloat;
use crate::sam::Sam;
use crate::utils::transforms::ResizeLongestSide;

pub struct SamPredictor<B: Backend> {
    is_image_set: bool,
    features: Option<Tensor<B, 4>>,
    orig_h: Option<i64>,
    orig_w: Option<i64>,
    input_h: Option<i64>,
    input_w: Option<i64>,
    input_size: Option<Size>,
    original_size: Option<Size>,
    pub model: Sam<B>,
    pub transfrom: ResizeLongestSide,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Module)]
pub enum ImageFormat {
    RGB,
    BGR,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Module)]
pub struct Size(pub usize, pub usize);

impl<B: Backend> SamPredictor<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    /// Uses SAM to calculate the image embedding for an image, and then
    /// allow repeated, efficient mask prediction given prompts.
    /// Arguments:
    ///   sam_model (Sam): The model to use for mask prediction.
    pub fn new(sam_model: Sam<B>) -> Self {
        let target_length = sam_model.image_encoder.img_size;
        SamPredictor {
            model: sam_model,
            transfrom: ResizeLongestSide::new(target_length),
            is_image_set: false,
            features: None,
            orig_h: None,
            orig_w: None,
            input_size: None,
            original_size: None,
            input_h: None,
            input_w: None,
        }
    }

    /// Calculates the image embeddings for the provided image, allowing
    ///     masks to be predicted with the 'predict' method.
    ///     Arguments:
    ///       image (np.ndarray): The image for calculating masks. Expects an
    ///         image in HWC uint8 format, with pixel values in [0, 255].
    ///       image_format (str): The color format of the image, in ['RGB', 'BGR'].
    pub fn set_image(&mut self, image: Tensor<B, 3, Int>, image_format: ImageFormat) {
        // assert!(
        //     image.kind() == ,
        //     "Image should be uint8, but is {:?}",
        //     image.kind()
        // );

        let image = if image_format != self.model.image_format {
            image.flip(vec![2]) // idk
        } else {
            image.clone()
        };

        let input_image = self.transfrom.apply_image(image.clone());
        let input_image = input_image.permute([2, 0, 1]).unsqueeze();
        let shape = image.dims();
        self.set_torch_image(input_image, Size(shape[0], shape[1]));
    }

    /// Calculates the image embeddings for the provided image, allowing
    /// masks to be predicted with the 'predict' method. Expects the input
    /// image to be already transformed to the format expected by the model.
    /// Arguments:
    ///   transformed_image (torch.Tensor): The input image, with shape
    ///     1x3xHxW, which has been transformed with ResizeLongestSide.
    ///   original_image_size (tuple(int, int)): The size of the image
    ///     before transformation, in (H, W) format.
    pub fn set_torch_image(&mut self, transformed_image: Tensor<B, 4, Int>, original_size: Size) {
        let shape = transformed_image.dims();
        assert!(
            shape.len() == 4
                && shape[1] == 3
                && shape[2..].iter().max().unwrap() == &self.model.image_encoder.img_size,
            "set_torch_image input must be BCHW with long side {}.",
            self.model.image_encoder.img_size,
        );
        self.original_size = Some(original_size);
        self.input_size = Some(Size(shape[2], shape[3]));
        let input_image = self.model.preprocess(transformed_image);
        let features = self.model.image_encoder.forward(input_image);
        self.features = Some(features);
        self.is_image_set = true;
    }

    /// Predict masks for the given input prompts, using the currently set image.
    /// Arguments:
    ///   point_coords (np.ndarray or None): A Nx2 array of point prompts to the
    ///     model. Each point is in (X,Y) in pixels.
    ///   point_labels (np.ndarray or None): A length N array of labels for the
    ///     point prompts. 1 indicates a foreground point and 0 indicates a
    ///     background point.
    ///   box (np.ndarray or None): A length 4 array given a box prompt to the
    ///     model, in XYXY format.
    ///   mask_input (np.ndarray): A low resolution mask input to the model, typically
    ///     coming from a previous prediction iteration. Has form 1xHxW, where
    ///     for SAM, H=W=256.
    ///   multimask_output (bool): If true, the model will return three masks.
    ///     For ambiguous input prompts (such as a single click), this will often
    ///     produce better masks than a single prediction. If only a single
    ///     mask is needed, the model's predicted quality score can be used
    ///     to select the best mask. For non-ambiguous prompts, such as multiple
    ///     input prompts, multimask_output=False can give better results.
    ///   return_logits (bool): If true, returns un-thresholded masks logits
    ///     instead of a binary mask.
    /// Returns:
    ///   (np.ndarray): The output masks in CxHxW format, where C is the
    ///     number of masks, and (H, W) is the original image size.
    ///   (np.ndarray): An array of length C containing the model's
    ///     predictions for the quality of each mask.
    ///   (np.ndarray): An array of shape CxHxW, where C is the number
    ///     of masks and H=W=256. These low resolution logits can be passed to
    ///     a subsequent iteration as mask input.
    pub fn predict(
        &self,
        point_coords: Option<Tensor<B, 2, Int>>,
        point_labels: Option<Tensor<B, 1, Int>>,
        boxes: Option<Tensor<B, 2, Int>>,
        mask_input: Option<Tensor<B, 3>>,
        multimask_output: bool,
    ) -> (Tensor<B, 3, Bool>, Tensor<B, 1>, Tensor<B, 3>) {
        // assert_eq!(
        //     &point_labels.unwrap().kind(),
        //     &Kind::Int,
        //     "point_labels must be int."
        // );
        assert!(self.is_image_set, "Must set image before predicting.");
        let (mut coords_torch, mut labels_torch, mut box_torch, mut mask_input_torch) =
            (None, None, None, None);
        if let Some(point_coords) = point_coords {
            let point_coords = self
                .transfrom
                .apply_coords(point_coords, self.original_size.unwrap());
            coords_torch = Some(point_coords.unsqueeze());
            labels_torch = Some(
                point_labels
                    .expect("point_labels must be supplied if point_coords is supplied.")
                    .to_float()
                    .unsqueeze(),
            );
        }
        if let Some(boxes) = boxes {
            let boxes = self
                .transfrom
                .apply_boxes(boxes, self.original_size.unwrap());
            box_torch = Some(boxes);
        }
        if let Some(mask_input) = mask_input {
            mask_input_torch = Some(mask_input.unsqueeze());
        }
        let (masks, iou_predictions, low_res_masks) = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
        );
        let masks = masks.select(0, 0);
        let iou_predictions = iou_predictions.select(0, 0);
        let low_res_masks = low_res_masks.select(0, 0);
        (masks, iou_predictions, low_res_masks)
    }
    /// Predict masks for the given input prompts, using the currently set image.
    /// Input prompts are batched torch tensors and are expected to already be
    /// transformed to the input frame using ResizeLongestSide.

    /// Arguments:
    ///   point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
    ///     model. Each point is in (X,Y) in pixels.
    ///   point_labels (torch.Tensor or None): A BxN array of labels for the
    ///     point prompts. 1 indicates a foreground point and 0 indicates a
    ///     background point.
    ///   boxes (np.ndarray or None): A Bx4 array given a box prompt to the
    ///     model, in XYXY format.
    ///   mask_input (np.ndarray): A low resolution mask input to the model, typically
    ///     coming from a previous prediction iteration. Has form Bx1xHxW, where
    ///     for SAM, H=W=256. Masks returned by a previous iteration of the
    ///     predict method do not need further transformation.
    ///   multimask_output (bool): If true, the model will return three masks.
    ///     For ambiguous input prompts (such as a single click), this will often
    ///     produce better masks than a single prediction. If only a single
    ///     mask is needed, the model's predicted quality score can be used
    ///     to select the best mask. For non-ambiguous prompts, such as multiple
    ///     input prompts, multimask_output=False can give better results.
    ///   return_logits (bool): If true, returns un-thresholded masks logits
    ///     instead of a binary mask.

    /// Returns:
    ///   (torch.Tensor): The output masks in BxCxHxW format, where C is the
    ///     number of masks, and (H, W) is the original image size.
    ///   (torch.Tensor): An array of shape BxC containing the model's
    ///     predictions for the quality of each mask.
    ///   (torch.Tensor): An array of shape BxCxHxW, where C is the number
    ///     of masks and H=W=256. These low res logits can be passed to
    ///     a subsequent iteration as mask input.
    pub fn predict_torch(
        &self,
        point_coords: Option<Tensor<B, 3>>,
        point_labels: Option<Tensor<B, 2>>,
        boxes: Option<Tensor<B, 2>>,
        mask_input: Option<Tensor<B, 4>>,
        multimask_output: bool,
    ) -> (Tensor<B, 4, Bool>, Tensor<B, 2>, Tensor<B, 4>) {
        assert!(self.is_image_set, "Must set image before predicting.");
        let point = match point_coords {
            Some(point_coords) => Some((point_coords, point_labels.unwrap())),
            None => None,
        };
        let (sparse_embeddings, dense_embeddings) =
            self.model.prompt_encoder.forward(point, boxes, mask_input);

        let (low_res_masks, iou_predictions) = self.model.mask_decoder.forward(
            self.features.clone().unwrap(),
            self.model.prompt_encoder.get_dense_pe(),
            sparse_embeddings,
            dense_embeddings,
            multimask_output,
        );
        let masks = self.model.postprocess_masks(
            low_res_masks.clone(),
            self.input_size.unwrap(),
            self.original_size.unwrap(),
        );
        let masks = masks.greater_elem(self.model.mask_threshold);
        return (masks, iou_predictions, low_res_masks);
    }
    /// Returns the image embeddings for the currently set image, with
    /// shape 1xCxHxW, where C is the embedding dimension and (H,W) are
    /// the embedding spatial dimension of SAM (typically C=256, H=W=64).
    pub fn get_image_embedding(&self) -> Tensor<B, 4> {
        assert!(self.is_image_set, "Must set image before predicting.");
        return self.features.clone().unwrap();
    }

    /// Resets the currently set image.
    pub fn reset_image(&mut self) {
        self.is_image_set = false;
        self.features = None;
        self.original_size = None;
        self.input_size = None;
        self.orig_h = None;
        self.orig_w = None;
        self.input_h = None;
        self.input_w = None;
    }
}

#[cfg(test)]
mod test {

    use crate::{
        build_sam::build_sam_test,
        tests::helpers::{random_tensor, random_tensor_int, Test, TestBackend, TEST_CHECKPOINT},
    };

    use super::{SamPredictor, Size};
    fn init(with_set_image: bool) -> SamPredictor<TestBackend> {
        let sam = build_sam_test(Some(TEST_CHECKPOINT));
        let mut predictor = SamPredictor::new(sam);
        if with_set_image {
            let image = random_tensor_int([120, 180, 3], 1, 255.);
            predictor.set_image(image, super::ImageFormat::RGB);
        }

        predictor
    }

    #[test]
    fn test_predictor_set_image() {
        let predictor = init(true);

        let file = Test::open("predictor_set_image");
        file.equal("original_size", predictor.original_size.unwrap());
        file.equal("input_size", predictor.input_size.unwrap());
        file.almost_equal("features", predictor.features.unwrap(),0.001);
        file.equal("is_image_set", predictor.is_image_set);
    }

    #[test]
    fn test_predictor_set_torch_image() {
        let mut predictor = init(false);

        let image = random_tensor_int([1, 3, 683, 1024], 1, 255.);
        let original_size = Size(120, 180);
        predictor.set_torch_image(image, original_size);
        let file = Test::open("predictor_set_torch_image");
        file.equal("original_size", predictor.original_size.unwrap());
        file.equal("input_size", predictor.input_size.unwrap());
        file.almost_equal("features", predictor.features.unwrap(),0.001);
        file.equal("is_image_set", predictor.is_image_set);
    }

    #[test]
    fn test_predictor_predict() {
        let predictor = init(true);

        let point_coords = random_tensor_int([1, 2], 1, 255.);
        let point_labels = random_tensor_int([1], 1, 1.);
        let (masks, iou_predictions, low_res_masks) =
            predictor.predict(Some(point_coords), Some(point_labels), None, None, true);
        let file = Test::open("predictor_predict");
        file.almost_equal("masks", masks,0.001);
        // file.compare("iou_predictions", iou_predictions);
        // file.compare("low_res_masks", low_res_masks);
    }

    #[test]
    fn test_predictor_predict_torch() {
        let predictor = init(true);

        let point_coords = random_tensor([1, 1, 2], 1);
        let point_labels = random_tensor([1, 1], 1);

        let (masks, iou_predictions, low_res_masks) =
            predictor.predict_torch(Some(point_coords), Some(point_labels), None, None, true);
        let file = Test::open("predictor_predict_torch");
        file.almost_equal("masks", masks,0.001);
        // file.compare("iou_predictions", iou_predictions); // Todo for some reason throwing
        // file.compare("low_res_masks", low_res_masks);
    }
}
