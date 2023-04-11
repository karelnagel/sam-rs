use tch::Tensor;

use crate::{
    modeling::{
        image_encoder::ImageEncoder, mask_decoder::MaskDecoder, prompt_encoder::PromptEncoder,
    },
    sam_predictor::Size,
};

pub struct Sam {
    pub image_encoder: ImageEncoder,
    pub prompt_encoder: PromptEncoder,
    pub mask_decoder: MaskDecoder,
}

impl Sam {
    pub fn preprocess(&self, image: Tensor) -> Tensor {
        // Todo
        image
    }
    pub fn postprocess_masks(
        &self,
        masks: &Tensor,
        input: &Option<Size>,
        original: &Option<Size>,
    ) -> Tensor {
        // Todo
        Tensor::zeros(&[1, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu))
    }
}
