use ndarray::{Array1, Array2, Array3};
use tch::{Kind, Tensor};

use crate::sam::Sam;
use crate::utils::transforms::ResizeLongestSide;

pub struct SamPredictor {
    is_image_set: bool,
    features: Option<Tensor>,
    orig_h: Option<i32>,
    orig_w: Option<i32>,
    input_h: Option<i32>,
    input_w: Option<i32>,
    input_size: Option<Size>,
    original_size: Option<Size>,
    device: Option<tch::Device>,
    model: Sam,
    transfrom: ResizeLongestSide,
}
pub enum ImageFormat {
    RGB,
    BGR,
}

#[derive(Clone, Copy)]
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
            device: None,
        }
    }

    /// Calculates the image embeddings for the provided image, allowing
    ///     masks to be predicted with the 'predict' method.
    ///     Arguments:
    ///       image (np.ndarray): The image for calculating masks. Expects an
    ///         image in HWC uint8 format, with pixel values in [0, 255].
    ///       image_format (str): The color format of the image, in ['RGB', 'BGR'].
    pub fn set_image(&mut self, image: Array3<u8>, image_format: &ImageFormat) {
        // Todo flip image if wrong format
        let input_image = self.transfrom.apply_image(&image);
        let input_image_tensor =
            Tensor::of_data_size(&input_image.into_raw_vec(), &[/* IDK */], Kind::Uint8);
        let mut input_image_torch = input_image_tensor.permute(&[2, 0, 1]);
        input_image_torch = input_image_torch.unsqueeze(0);
        if let Some(device) = self.device {
            input_image_torch = input_image_torch.to_device(device);
        }
        let shape = image.shape();
        self.set_torch_image(input_image_torch, Size(shape[0] as i64, shape[1] as i64));
    }

    /// Calculates the image embeddings for the provided image, allowing
    /// masks to be predicted with the 'predict' method. Expects the input
    /// image to be already transformed to the format expected by the model.
    /// Arguments:
    ///   transformed_image (torch.Tensor): The input image, with shape
    ///     1x3xHxW, which has been transformed with ResizeLongestSide.
    ///   original_image_size (tuple(int, int)): The size of the image
    ///     before transformation, in (H, W) format.
    pub fn set_torch_image(&mut self, transformed_image: Tensor, original_size: Size) {
        // Todo apply @torch.no_grad()
        let shape = transformed_image.size();
        if shape.len() != 4
            || shape[1] != 3
            || *shape[2..].iter().max().unwrap() != self.model.image_encoder.img_size.into()
        {
            panic!(
                "set_torch_image input must be BCHW with long side {}.",
                self.model.image_encoder.img_size
            );
        }
        self.original_size = Some(original_size);
        self.input_size = Some(Size(shape[2], shape[3]));

        let input_image = self.model.preprocess(transformed_image);
        self.features = Some(self.model.image_encoder.forward(&input_image));
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
        point_coords: Option<Array2<f32>>,
        point_labels: Option<Array1<i32>>,
        boxes: Option<Array1<f32>>,
        mask_input: Option<Array1<f32>>,
        multimask_output: bool,
        return_logits: bool,
    ) -> (Tensor, Tensor, Tensor) {
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
                .apply_coords(&point_coords, &self.original_size.unwrap());
            coords_torch = Some(Tensor::of_slice(&point_coords.into_raw_vec()));
            labels_torch = Some(Tensor::of_slice(&point_labels.unwrap().to_vec()));
        }
        if let Some(mut boxes) = boxes {
            boxes = self
                .transfrom
                .apply_boxes(&boxes, &self.original_size.unwrap());
            box_torch = Some(Tensor::of_slice(&boxes.to_vec()));
        }
        if let Some(mask_input) = mask_input {
            mask_input_torch = Some(Tensor::of_slice(&mask_input.to_vec()));
        }
        let (masks, iou_predictions, low_res_masks) = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits,
        );
        let masks = masks.to_kind(Kind::Float);
        let iou_predictions = iou_predictions.to_kind(Kind::Float);
        let low_res_masks = low_res_masks.to_kind(Kind::Float);
        return (masks, iou_predictions, low_res_masks);
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
        point_coords: Option<Tensor>,
        point_labels: Option<Tensor>,
        boxes: Option<Tensor>,
        mask_input: Option<Tensor>,
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
            self.model.prompt_encoder.get_dense_pe(),
            sparse_embeddings,
            dense_embeddings,
            multimask_output,
        );
        let masks = self.model.postprocess_masks(
            &low_res_masks,
            &self.input_size.unwrap(),
            &self.original_size.unwrap(),
        );
        if !return_logits {
            panic!("Not implemented");
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
