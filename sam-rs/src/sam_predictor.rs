use serde::{Deserialize, Serialize};
use tch::{Kind, Tensor};

use crate::sam::Sam;
use crate::utils::transforms::ResizeLongestSide;

pub struct SamPredictor {
    is_image_set: bool,
    features: Option<Tensor>,
    orig_h: Option<i64>,
    orig_w: Option<i64>,
    input_h: Option<i64>,
    input_w: Option<i64>,
    input_size: Option<Size>,
    original_size: Option<Size>,
    model: Sam,
    transfrom: ResizeLongestSide,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    RGB,
    BGR,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Size(pub i64, pub i64);

impl SamPredictor {
    /// Uses SAM to calculate the image embedding for an image, and then
    /// allow repeated, efficient mask prediction given prompts.
    /// Arguments:
    ///   sam_model (Sam): The model to use for mask prediction.
    pub fn new(sam_model: Sam) -> SamPredictor {
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
    pub fn set_image(&mut self, image: &Tensor, image_format: ImageFormat) {
        assert!(
            image.kind() == Kind::Uint8,
            "Image should be uint8, but is {:?}",
            image.kind()
        );

        let image = if image_format != self.model.image_format {
            image.flip(&[-1])
        } else {
            image.copy()
        };

        let mut input_image = self.transfrom.apply_image(&image);
        input_image = input_image.permute(&[2, 0, 1]).unsqueeze(0);
        let shape = image.size();
        self.set_torch_image(&input_image, Size(shape[0] as i64, shape[1] as i64));
    }

    /// Calculates the image embeddings for the provided image, allowing
    /// masks to be predicted with the 'predict' method. Expects the input
    /// image to be already transformed to the format expected by the model.
    /// Arguments:
    ///   transformed_image (torch.Tensor): The input image, with shape
    ///     1x3xHxW, which has been transformed with ResizeLongestSide.
    ///   original_image_size (tuple(int, int)): The size of the image
    ///     before transformation, in (H, W) format.
    pub fn set_torch_image(&mut self, transformed_image: &Tensor, original_size: Size) {
        let shape = transformed_image.size();
        assert!(
            shape.len() == 4
                && shape[1] == 3
                && shape[2..].iter().max().unwrap() == &self.model.image_encoder.img_size,
            "set_torch_image input must be BCHW with long side {}.",
            self.model.image_encoder.img_size,
        );
        self.original_size = Some(original_size);
        self.input_size = Some(Size(shape[2], shape[3]));
        let input_image = self.model.preprocess(&transformed_image);
        let features = self.model.image_encoder.forward(&input_image);
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
        point_coords: Option<&Tensor>,
        point_labels: Option<&Tensor>,
        boxes: Option<&Tensor>,
        mask_input: Option<&Tensor>,
        multimask_output: bool,
        return_logits: bool,
    ) -> (Tensor, Tensor, Tensor) {
        assert_eq!(
            &point_labels.unwrap().kind(),
            &Kind::Int,
            "point_labels must be int."
        );
        assert!(self.is_image_set, "Must set image before predicting.");
        let (mut coords_torch, mut labels_torch, mut box_torch, mut mask_input_torch) =
            (None, None, None, None);
        if let Some(point_coords) = point_coords {
            assert!(
                point_labels.is_some(),
                "point_labels must be supplied if point_coords is supplied."
            );
            let point_coords = self
                .transfrom
                .apply_coords(&point_coords, self.original_size.unwrap());
            coords_torch = Some(point_coords.unsqueeze(0));
            labels_torch = Some(point_labels.unwrap().unsqueeze(0));
        }
        if let Some(boxes) = boxes {
            let boxes = self
                .transfrom
                .apply_boxes(&boxes, self.original_size.unwrap());
            box_torch = Some(boxes.unsqueeze(0));
        }
        if let Some(mask_input) = mask_input {
            mask_input_torch = Some(mask_input.unsqueeze(0));
        }
        let (masks, iou_predictions, low_res_masks) = self.predict_torch(
            coords_torch.as_ref(),
            labels_torch.as_ref(),
            box_torch.as_ref(),
            mask_input_torch.as_ref(),
            multimask_output,
            return_logits,
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
        point_coords: Option<&Tensor>,
        point_labels: Option<&Tensor>,
        boxes: Option<&Tensor>,
        mask_input: Option<&Tensor>,
        multimask_output: bool,
        return_logits: bool,
    ) -> (Tensor, Tensor, Tensor) {
        assert!(self.is_image_set, "Must set image before predicting.");
        let point = match point_coords {
            Some(point_coords) => Some((point_coords, point_labels.unwrap())),
            None => None,
        };
        let (sparse_embeddings, dense_embeddings) =
            self.model.prompt_encoder.forward(point, boxes, mask_input);

        let (low_res_masks, iou_predictions) = self.model.mask_decoder.forward(
            self.features.as_ref().unwrap(),
            &self.model.prompt_encoder.get_dense_pe(),
            &sparse_embeddings,
            &dense_embeddings,
            multimask_output,
        );
        let mut masks = self.model.postprocess_masks(
            &low_res_masks,
            &self.input_size.unwrap(),
            &self.original_size.unwrap(),
        );
        if !return_logits {
            masks = masks.gt(self.model.mask_threshold)
        }
        return (masks, iou_predictions, low_res_masks);
    }
    /// Returns the image embeddings for the currently set image, with
    /// shape 1xCxHxW, where C is the embedding dimension and (H,W) are
    /// the embedding spatial dimension of SAM (typically C=256, H=W=64).
    pub fn get_image_embedding(&self) -> &Tensor {
        assert!(self.is_image_set, "Must set image before predicting.");
        return self.features.as_ref().unwrap();
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
        build_sam::build_sam_vit_b,
        tests::{
            helpers::{random_tensor, TestFile},
            mocks::Mock,
        },
    };

    use super::{SamPredictor, Size};

    fn init(with_set_image: bool) -> SamPredictor {
        let mut sam = build_sam_vit_b(None);
        sam.mock();
        let mut predictor = SamPredictor::new(sam);
        if with_set_image {
            let image = (random_tensor(&[120, 180, 3], 1) * 255).to_kind(tch::Kind::Uint8);
            predictor.set_image(&image, super::ImageFormat::RGB);
        }

        predictor
    }

    #[test]
    fn test_predictor_set_image() {
        let predictor = init(true);

        let file = TestFile::open("predictor_set_image");
        file.compare("original_size", predictor.original_size.unwrap());
        file.compare("input_size", predictor.input_size.unwrap());
        file.compare("features", predictor.features.unwrap());
        file.compare("is_image_set", predictor.is_image_set);
    }

    #[test]
    fn test_predictor_set_torch_image() {
        let mut predictor = init(false);

        let image = random_tensor(&[1, 3, 683, 1024], 1);
        let original_size = Size(120, 180);
        predictor.set_torch_image(&image, original_size);
        let file = TestFile::open("predictor_set_torch_image");
        file.compare("original_size", predictor.original_size.unwrap());
        file.compare("input_size", predictor.input_size.unwrap());
        file.compare("features", predictor.features.unwrap());
        file.compare("is_image_set", predictor.is_image_set);
    }

    #[test]
    fn test_predictor_predict() {
        let predictor = init(true);

        let point_coords = random_tensor(&[1, 2], 1);
        let point_labels = (random_tensor(&[1], 1) * 255).to_kind(tch::Kind::Int);
        let (masks, iou_predictions, low_res_masks) = predictor.predict(
            Some(&point_coords),
            Some(&point_labels),
            None,
            None,
            true,
            false,
        );
        let file = TestFile::open("predictor_predict");
        file.compare("masks", masks);
        file.compare("iou_predictions", iou_predictions);
        file.compare("low_res_masks", low_res_masks);
    }

    #[test]
    fn test_predictor_predict_torch() {
        let predictor = init(true);

        let point_coords = random_tensor(&[1, 1, 2], 1);
        let point_labels = random_tensor(&[1, 1], 1);

        let (masks, iou_predictions, low_res_masks) = predictor.predict_torch(
            Some(&point_coords),
            Some(&point_labels),
            None,
            None,
            true,
            false,
        );
        let file = TestFile::open("predictor_predict_torch");
        file.compare("masks", masks);
        file.compare("iou_predictions", iou_predictions);
        file.compare("low_res_masks", low_res_masks);
    }
}
