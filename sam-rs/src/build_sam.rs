use burn::{module::Module, record::SentitiveCompactRecordSettings, tensor::backend::Backend};
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
pub enum BuildSam {
    SamVitH,
    SamVitL,
    SamVitB,
    SamTest,
}
impl BuildSam {
    pub fn build<B: Backend>(&self, checkpoint: Option<&str>) -> Sam<B>
    where
        <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
    {
        match self {
            Self::SamVitH => build_sam_vit_h(checkpoint),
            Self::SamVitL => build_sam_vit_l(checkpoint),
            Self::SamVitB => build_sam_vit_b(checkpoint),
            Self::SamTest => build_sam_test(checkpoint),
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
    let activation = Activation::GELU;
    let activation_relu = Activation::ReLU;
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
            activation_relu,
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
            Some(activation),
        ),
        MaskDecoder::new(
            prompt_embed_dim,
            TwoWayTransformer::new(2, prompt_embed_dim, 8, 2048, Some(activation), None),
            Some(3),
            Some(activation),
            Some(3),
            Some(256),
        ),
        Some([123.675, 116.28, 103.53]),
        Some([58.395, 57.12, 57.375]),
    );
    if let Some(checkpoint) = _checkpoint {
        let record =
            burn::record::Record::load::<SentitiveCompactRecordSettings>(checkpoint.into())
                .unwrap();
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
        file.compare("mask_threshold", sam.mask_threshold);
        file.compare("image_format", sam.image_format);
        file.compare(
            "mask_decoder.num_mask_tokens",
            sam.mask_decoder.num_mask_tokens,
        );
        file.compare("prompt_encoder.embed_dim", sam.prompt_encoder.embed_dim);
        file.compare(
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
