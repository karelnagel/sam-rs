use tch::{nn::VarStore, Device};

use crate::{
    modeling::{
        image_encoder::ImageEncoderViT,
        mask_decoder::{Activation, MaskDecoder},
        prompt_encoder::PromptEncoder,
        transformer::TwoWayTransformer,
    },
    sam::Sam,
    sam_predictor::Size,
};

pub fn build_sam_vit_h(checkpoint: Option<&str>) -> Sam {
    _build_sam(1280, 32, 16, [7, 15, 23, 31], checkpoint)
}

pub fn build_sam_vit_l(checkpoint: Option<&str>) -> Sam {
    _build_sam(1024, 24, 16, [5, 11, 17, 23], checkpoint)
}
pub fn build_sam_vit_b(checkpoint: Option<&str>) -> Sam {
    _build_sam(768, 12, 12, [2, 5, 8, 11], checkpoint)
}

fn _build_sam(
    encoder_embed_dim: i64,
    encoder_depth: i64,
    encoder_num_heads: i64,
    encoder_global_attn_indexes: [i64; 4],
    checkpoint: Option<&str>,
) -> Sam {
    let prompt_embed_dim = 256;
    let img_size = 1024;
    let vit_patch_size = 16;
    let image_embedding_size = img_size; // vit_patch_size
    let vs = VarStore::new(Device::cuda_if_available());
    let vs = &vs.root();
    let sam = Sam::new(
        ImageEncoderViT::new(
            vs,
            Some(img_size),
            Some(vit_patch_size),
            None,
            Some(encoder_embed_dim),
            Some(encoder_depth),
            Some(encoder_num_heads),
            Some(4.0),
            Some(prompt_embed_dim),
            Some(true),
            None,
            None,
            Some(true),
            None,
            Some(14),
            Some(&encoder_global_attn_indexes),
        ),
        PromptEncoder::new(
            vs,
            prompt_embed_dim,
            Size(image_embedding_size, image_embedding_size),
            Size(img_size, img_size),
            16,
            Activation::GELU,
        ),
        MaskDecoder::new(
            vs,
            prompt_embed_dim,
            TwoWayTransformer::new(
                vs,
                2,
                prompt_embed_dim,
                8,
                2048,
                Some(Activation::GELU),
                None,
            ),
            3,
            Activation::GELU,
            3,
            256,
        ),
        Some(&[123.675, 116.28, 103.53]),
        Some(&[58.395, 57.12, 57.375]),
    );
    
    // Todo
    // if checkpoint is not None:
    //     with open(checkpoint, "rb") as f:
    //         state_dict = torch.load(f)
    //     sam.load_state_dict(state_dict)
    // return sam
    sam
}
