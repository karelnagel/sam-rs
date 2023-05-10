use burn::{
    module::Module,
    record::{BinGzFileRecorder, DoublePrecisionSettings, Recorder},
    tensor::backend::Backend,
};
use serde::{Deserialize, Serialize};

use crate::{
    modeling::{
        common::activation::Activation, image_encoder::ImageEncoderViT, mask_decoder::MaskDecoder,
        prompt_encoder::PromptEncoder, transformer::TwoWayTransformer,
    },
    sam::Sam,
    sam_predictor::Size,
};
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum SamVersion {
    VitH,
    VitL,
    VitB,
    Test,
}
impl SamVersion {
    pub fn build<B: Backend>(&self, checkpoint: Option<&str>) -> Sam<B>
    where
        <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
    {
        match self {
            Self::VitH => build_sam_vit_h(checkpoint),
            Self::VitL => build_sam_vit_l(checkpoint),
            Self::VitB => build_sam_vit_b(checkpoint),
            Self::Test => build_sam_test(checkpoint),
        }
    }
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::VitH => "vit_h",
            Self::VitL => "vit_l",
            Self::VitB => "vit_b",
            Self::Test => "test",
        }
    }
    pub fn from_str(s: &str) -> Self {
        match s {
            "vit_h" => Self::VitH,
            "vit_l" => Self::VitL,
            "vit_b" => Self::VitB,
            "test" => Self::Test,
            _ => panic!("Unknown variant: {}", s),
        }
    }
}
pub fn build_sam_vit_h<B: Backend>(checkpoint: Option<&str>) -> Sam<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    _build_sam(1280, 32, 16, vec![7, 15, 23, 31], checkpoint)
}

pub fn build_sam_vit_l<B: Backend>(checkpoint: Option<&str>) -> Sam<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    _build_sam(1024, 24, 16, vec![5, 11, 17, 23], checkpoint)
}
pub fn build_sam_vit_b<B: Backend>(checkpoint: Option<&str>) -> Sam<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    _build_sam(768, 12, 12, vec![2, 5, 8, 11], checkpoint)
}

pub fn build_sam_test<B: Backend>(checkpoint: Option<&str>) -> Sam<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    _build_sam(8, 2, 2, vec![2, 5, 8, 11], checkpoint)
}

fn _build_sam<B: Backend>(
    encoder_embed_dim: usize,
    encoder_depth: usize,
    encoder_num_heads: usize,
    encoder_global_attn_indexes: Vec<usize>,
    _checkpoint: Option<&str>,
) -> Sam<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    let prompt_embed_dim = 256;
    let img_size = 1024;
    let vit_patch_size = 16;
    let image_embedding_size = img_size / vit_patch_size;
    let mut sam = Sam::new(
        ImageEncoderViT::new(
            Some(img_size),
            Some(vit_patch_size),
            None,
            Some(encoder_embed_dim),
            Some(encoder_depth),
            Some(encoder_num_heads),
            Some(4.0),
            Some(prompt_embed_dim),
            Some(true),
            Activation::GELU,
            None,
            Some(true),
            None,
            Some(14),
            Some(encoder_global_attn_indexes),
        ),
        PromptEncoder::new(
            prompt_embed_dim,
            Size(image_embedding_size, image_embedding_size),
            Size(img_size, img_size),
            16,
            None,
        ),
        MaskDecoder::new(
            prompt_embed_dim,
            TwoWayTransformer::new(2, prompt_embed_dim, 8, 2048, None, None),
            Some(3),
            None,
            Some(3),
            Some(256),
        ),
        Some([123.675, 116.28, 103.53]),
        Some([58.395, 57.12, 57.375]),
    );
    if let Some(checkpoint) = _checkpoint {
        let recorder = BinGzFileRecorder::<DoublePrecisionSettings>::default();
        let record = recorder.load(checkpoint.into()).unwrap();
        sam = sam.load_record(record);
    }
    sam
}

#[cfg(test)]
mod test {
    use burn::tensor::backend::Backend;

    use crate::tests::helpers::{Test, TestBackend};

    fn test<B: Backend>(name: &str, sam: crate::sam::Sam<B>) {
        let file = Test::open(name);
        file.equal("mask_threshold", sam.mask_threshold);
        file.equal("image_format", sam.image_format);
        file.equal(
            "mask_decoder.num_mask_tokens",
            sam.mask_decoder.num_mask_tokens,
        );
        file.equal("prompt_encoder.embed_dim", sam.prompt_encoder.embed_dim);
        file.equal(
            "prompt_encoder.input_image_size",
            sam.prompt_encoder.input_image_size,
        );
    }
    #[test]
    fn test_build_sam_vit_h() {
        let sam = super::build_sam_vit_h::<TestBackend>(None);
        test("sam_vit_h", sam);
    }
    #[test]
    fn test_build_sam_vit_l() {
        let sam = super::build_sam_vit_l::<TestBackend>(None);
        test("sam_vit_l", sam);
    }
    #[test]
    fn test_build_sam_vit_b() {
        let sam = super::build_sam_vit_b::<TestBackend>(None);
        test("sam_vit_b", sam);
    }
}
