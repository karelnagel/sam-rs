use std::borrow::BorrowMut;

use crate::{python::python_data::pyany_to_tensor, sam::SamRecord};
use burn::{
    module::Param,
    tensor::{backend::Backend, Tensor},
};
use pyo3::PyAny;

pub fn _print_match_key(key: &str) {
    match key.contains("bias") || key.contains("rel_pos_h") || key.contains("rel_pos_w") {
        false => println!("\"{key}\" => set_value(sam.{key}.borrow_mut(),value,key),"),
        true => println!("\"{key}\" => set_value_opt(sam.{key}.borrow_mut(),value,key),"),
    }
}

const TRANSPOSED: [&str; 7] = [
    "lin1.weight",
    "lin2.weight",
    "qkv.weight",
    "proj.weight",
    "gamma.weight",
    "stride.weight",
    "padding.weight",
];
fn needs_transpose(key: &str) -> bool {
    TRANSPOSED.iter().any(|x| key.contains(x)) || (key.contains("layers") && key.contains("weight"))
}
fn set_value<B: Backend, const D: usize>(old: &mut Param<Tensor<B, D>>, new: &PyAny, key: &str) {
    let mut new = pyany_to_tensor(new);
    if needs_transpose(key) {
        new = new.transpose();
    }
    assert_eq!(old.dims(), new.dims(), "Dims not same for: {key}");
    *old = Param::from(new);
}
fn set_value_opt<B: Backend, const D: usize>(
    old: &mut Option<Param<Tensor<B, D>>>,
    new: &PyAny,
    key: &str,
) {
    let mut new = pyany_to_tensor(new);
    if needs_transpose(key) {
        new = new.transpose();
    }
    assert_eq!(
        old.as_ref()
            .expect(format!("There is no tensor for {key}").as_str())
            .dims(),
        new.dims(),
        "Dims not same for: {key}"
    );
    *old = Some(Param::from(new));
}

pub fn update_tensor<B: Backend>(sam: &mut SamRecord<B>, key: &str, value: &PyAny) {
    match key {
        "image_encoder.blocks[6].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[6].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].norm1.gamma" => set_value(
            sam.image_encoder.blocks[0].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[10].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[9].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_token_to_image.q_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_token_to_image
                    .q_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.blocks[4].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[4].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[5].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_upscaling1.weight" => set_value(
            sam.mask_decoder.output_upscaling1.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[3].layers[0].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[3].layers[0]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[2].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[3].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.iou_prediction_head.layers[1].bias" => set_value_opt(
            sam.mask_decoder.iou_prediction_head.layers[1]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.norm_final_attn.gamma" => set_value(
            sam.mask_decoder
                .transformer
                .norm_final_attn
                .gamma
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[9].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[4].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].norm1.gamma" => set_value(
            sam.image_encoder.blocks[3].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[5].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].norm1.beta" => set_value(
            sam.image_encoder.blocks[8].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].mlp.lin1.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[0]
                .mlp
                .lin1
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling0.bias" => set_value_opt(
            sam.prompt_encoder.mask_downscaling0.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_token_to_image.q_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_token_to_image
                    .q_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.blocks[1].norm2.beta" => set_value(
            sam.image_encoder.blocks[1].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].norm1.beta" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .norm1
                .beta
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].norm1.gamma" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .norm1
                .gamma
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[11].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.final_attn_token_to_image.q_proj.weight" => set_value(
            sam.mask_decoder
                .transformer
                .final_attn_token_to_image
                .q_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[9].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.final_attn_token_to_image.k_proj.weight" => set_value(
            sam.mask_decoder
                .transformer
                .final_attn_token_to_image
                .k_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_upscaling1.bias" => set_value(
            sam.mask_decoder.output_upscaling1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[5].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].norm1.gamma" => set_value(
            sam.image_encoder.blocks[8].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[9].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].norm2.gamma" => set_value(
            sam.image_encoder.blocks[5].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].norm1.beta" => set_value(
            sam.image_encoder.blocks[7].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].norm2.beta" => set_value(
            sam.image_encoder.blocks[2].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[5].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[2].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[6].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].mlp.lin2.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .mlp
                .lin2
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[10].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].norm4.beta" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .norm4
                .beta
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[1].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[6].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[7].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].self_attn.out_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .self_attn
                .out_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.neck2.weight" => {
            set_value(sam.image_encoder.neck2.weight.borrow_mut(), value, key)
        }
        "mask_decoder.output_hypernetworks_mlps[1].layers[1].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[1].layers[1]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[2].layers[1].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[2].layers[1]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[0].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[2].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[7].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[11].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[3].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.final_attn_token_to_image.v_proj.bias" => set_value_opt(
            sam.mask_decoder
                .transformer
                .final_attn_token_to_image
                .v_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].norm3.beta" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .norm3
                .beta
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[3].layers[1].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[3].layers[1]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[8].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[6].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].norm2.gamma" => set_value(
            sam.image_encoder.blocks[10].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling4.weight" => set_value(
            sam.prompt_encoder.mask_downscaling4.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_token_to_image.v_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_token_to_image
                    .v_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.transformer.layers[1].cross_attn_image_to_token.k_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_image_to_token
                    .k_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.blocks[5].norm2.beta" => set_value(
            sam.image_encoder.blocks[5].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[8].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_upscaling3.bias" => set_value_opt(
            sam.mask_decoder.output_upscaling3.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[4].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[6].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[2].layers[0].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[2].layers[0]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].norm2.gamma" => set_value(
            sam.image_encoder.blocks[11].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.point_embeddings[0].weight" => set_value(
            sam.prompt_encoder.point_embeddings[0].weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].norm2.beta" => set_value(
            sam.image_encoder.blocks[9].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_token_to_image.k_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_token_to_image
                    .k_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.blocks[5].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[5].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[7].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[6].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[8].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].norm1.gamma" => set_value(
            sam.image_encoder.blocks[9].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[9].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling6.weight" => set_value(
            sam.prompt_encoder.mask_downscaling6.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].self_attn.q_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .self_attn
                .q_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].norm2.gamma" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .norm2
                .gamma
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[1].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[3].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[1].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[2].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[11].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].norm2.gamma" => set_value(
            sam.image_encoder.blocks[6].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_image_to_token.k_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .cross_attn_image_to_token
                .k_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[4].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].mlp.lin2.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[1]
                .mlp
                .lin2
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].self_attn.q_proj.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[1]
                .self_attn
                .q_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[1].layers[2].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[1].layers[2]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[2].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_image_to_token.q_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_image_to_token
                    .q_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.output_hypernetworks_mlps[1].layers[0].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[1].layers[0]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[1].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].norm2.beta" => set_value(
            sam.image_encoder.blocks[6].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].self_attn.v_proj.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[0]
                .self_attn
                .v_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[2].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].norm1.gamma" => set_value(
            sam.image_encoder.blocks[5].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.final_attn_token_to_image.out_proj.weight" => set_value(
            sam.mask_decoder
                .transformer
                .final_attn_token_to_image
                .out_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[2].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].self_attn.k_proj.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[0]
                .self_attn
                .k_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].norm1.beta" => set_value(
            sam.image_encoder.blocks[1].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[1].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[11].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].norm2.gamma" => set_value(
            sam.image_encoder.blocks[3].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[2].layers[1].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[2].layers[1]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].norm2.gamma" => set_value(
            sam.image_encoder.blocks[0].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling1.weight" => set_value(
            sam.prompt_encoder.mask_downscaling1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].norm2.gamma" => set_value(
            sam.image_encoder.blocks[2].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].norm1.gamma" => set_value(
            sam.image_encoder.blocks[2].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].norm1.gamma" => set_value(
            sam.image_encoder.blocks[1].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[1].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_token_to_image.k_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .cross_attn_token_to_image
                .k_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[1].layers[0].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[1].layers[0]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[5].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].norm1.beta" => set_value(
            sam.image_encoder.blocks[6].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_token_to_image.q_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .cross_attn_token_to_image
                .q_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling0.weight" => set_value(
            sam.prompt_encoder.mask_downscaling0.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[0].layers[1].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[0].layers[1]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_token_to_image.out_proj.weight" => {
            set_value(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_token_to_image
                    .out_proj
                    .weight
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.blocks[4].norm1.gamma" => set_value(
            sam.image_encoder.blocks[4].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].norm1.gamma" => set_value(
            sam.image_encoder.blocks[7].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.neck0.weight" => {
            set_value(sam.image_encoder.neck0.weight.borrow_mut(), value, key)
        }
        "prompt_encoder.point_embeddings[3].weight" => set_value(
            sam.prompt_encoder.point_embeddings[3].weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_image_to_token.out_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_image_to_token
                    .out_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.output_hypernetworks_mlps[3].layers[0].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[3].layers[0]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].norm2.beta" => set_value(
            sam.image_encoder.blocks[11].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.point_embeddings[1].weight" => set_value(
            sam.prompt_encoder.point_embeddings[1].weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[0].layers[1].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[0].layers[1]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].norm2.beta" => set_value(
            sam.image_encoder.blocks[10].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].norm2.gamma" => set_value(
            sam.image_encoder.blocks[9].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_token_to_image.v_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .cross_attn_token_to_image
                .v_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[3].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].self_attn.k_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .self_attn
                .k_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[3].layers[2].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[3].layers[2]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[1].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[0].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].norm1.gamma" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .norm1
                .gamma
                .borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling3.weight" => set_value(
            sam.prompt_encoder.mask_downscaling3.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[11].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[9].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[7].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_image_to_token.q_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .cross_attn_image_to_token
                .q_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].norm3.gamma" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .norm3
                .gamma
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].norm4.beta" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .norm4
                .beta
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[0].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[0].layers[0].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[0].layers[0]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[7].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[0].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_token_to_image.k_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_token_to_image
                    .k_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.blocks[6].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[6].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.final_attn_token_to_image.k_proj.bias" => set_value_opt(
            sam.mask_decoder
                .transformer
                .final_attn_token_to_image
                .k_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_image_to_token.v_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .cross_attn_image_to_token
                .v_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[3].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[2].layers[0].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[2].layers[0]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[5].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_image_to_token.q_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .cross_attn_image_to_token
                .q_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].norm4.gamma" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .norm4
                .gamma
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_image_to_token.out_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_image_to_token
                    .out_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.blocks[11].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[11].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_token_to_image.v_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_token_to_image
                    .v_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.output_hypernetworks_mlps[3].layers[2].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[3].layers[2]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].norm2.beta" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .norm2
                .beta
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].norm1.beta" => set_value(
            sam.image_encoder.blocks[2].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[4].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].mlp.lin1.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[1]
                .mlp
                .lin1
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[10].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.neck1.bias" => {
            set_value(sam.image_encoder.neck1.bias.borrow_mut(), value, key)
        }
        "mask_decoder.transformer.norm_final_attn.beta" => set_value(
            sam.mask_decoder
                .transformer
                .norm_final_attn
                .beta
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].norm1.beta" => set_value(
            sam.image_encoder.blocks[3].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.point_embeddings[2].weight" => set_value(
            sam.prompt_encoder.point_embeddings[2].weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[3].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[8].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[1].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[2].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].norm2.beta" => set_value(
            sam.image_encoder.blocks[7].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[10].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling6.bias" => set_value_opt(
            sam.prompt_encoder.mask_downscaling6.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].self_attn.out_proj.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[1]
                .self_attn
                .out_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[2].layers[2].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[2].layers[2]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[7].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[10].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.iou_prediction_head.layers[0].weight" => set_value(
            sam.mask_decoder.iou_prediction_head.layers[0]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].mlp.lin2.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[0]
                .mlp
                .lin2
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[7].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].norm1.gamma" => set_value(
            sam.image_encoder.blocks[6].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling1.bias" => set_value(
            sam.prompt_encoder.mask_downscaling1.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].self_attn.q_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .self_attn
                .q_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_upscaling0.bias" => set_value_opt(
            sam.mask_decoder.output_upscaling0.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[8].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.no_mask_embed.weight" => set_value(
            sam.prompt_encoder.no_mask_embed.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[7].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[8].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_token_to_image.v_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .cross_attn_token_to_image
                .v_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].mlp.lin1.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .mlp
                .lin1
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_image_to_token.out_proj.weight" => {
            set_value(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_image_to_token
                    .out_proj
                    .weight
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.blocks[5].norm1.beta" => set_value(
            sam.image_encoder.blocks[5].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[0].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].self_attn.out_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .self_attn
                .out_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[4].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[8].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[8].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].mlp.lin2.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .mlp
                .lin2
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[10].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_token_to_image.k_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .cross_attn_token_to_image
                .k_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.final_attn_token_to_image.q_proj.bias" => set_value_opt(
            sam.mask_decoder
                .transformer
                .final_attn_token_to_image
                .q_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[10].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.mask_tokens.weight" => {
            set_value(sam.mask_decoder.mask_tokens.weight.borrow_mut(), value, key)
        }
        "mask_decoder.iou_prediction_head.layers[2].weight" => set_value(
            sam.mask_decoder.iou_prediction_head.layers[2]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[9].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.patch_embed.proj.bias" => set_value_opt(
            sam.image_encoder.patch_embed.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[4].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.pos_embed" => {
            set_value_opt(sam.image_encoder.pos_embed.borrow_mut(), value, key)
        }
        "mask_decoder.transformer.layers[0].self_attn.q_proj.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[0]
                .self_attn
                .q_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].norm2.gamma" => set_value(
            sam.image_encoder.blocks[4].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_image_to_token.v_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_image_to_token
                    .v_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.transformer.final_attn_token_to_image.out_proj.bias" => set_value_opt(
            sam.mask_decoder
                .transformer
                .final_attn_token_to_image
                .out_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.iou_token.weight" => {
            set_value(sam.mask_decoder.iou_token.weight.borrow_mut(), value, key)
        }
        "mask_decoder.output_hypernetworks_mlps[3].layers[1].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[3].layers[1]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].norm2.beta" => set_value(
            sam.image_encoder.blocks[3].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[2].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].norm2.gamma" => set_value(
            sam.image_encoder.blocks[8].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].norm2.beta" => set_value(
            sam.image_encoder.blocks[4].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_image_to_token.k_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .cross_attn_image_to_token
                .k_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.neck3.weight" => {
            set_value(sam.image_encoder.neck3.weight.borrow_mut(), value, key)
        }
        "mask_decoder.output_hypernetworks_mlps[0].layers[2].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[0].layers[2]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.iou_prediction_head.layers[1].weight" => set_value(
            sam.mask_decoder.iou_prediction_head.layers[1]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[4].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[3].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].norm2.gamma" => set_value(
            sam.image_encoder.blocks[1].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling3.bias" => set_value_opt(
            sam.prompt_encoder.mask_downscaling3.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[0].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[11].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].norm3.gamma" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .norm3
                .gamma
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].norm3.beta" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .norm3
                .beta
                .borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.not_a_point_embed.weight" => set_value(
            sam.prompt_encoder.not_a_point_embed.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].norm1.beta" => set_value(
            sam.image_encoder.blocks[10].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[10].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].self_attn.v_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .self_attn
                .v_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.iou_prediction_head.layers[2].bias" => set_value_opt(
            sam.mask_decoder.iou_prediction_head.layers[2]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_upscaling3.weight" => set_value(
            sam.mask_decoder.output_upscaling3.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[2].layers[2].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[2].layers[2]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].self_attn.out_proj.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[0]
                .self_attn
                .out_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].norm2.gamma" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .norm2
                .gamma
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[3].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].self_attn.v_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .self_attn
                .v_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[8].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[1].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[9].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[11].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].self_attn.k_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .self_attn
                .k_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.neck1.weight" => {
            set_value(sam.image_encoder.neck1.weight.borrow_mut(), value, key)
        }
        "mask_decoder.output_hypernetworks_mlps[1].layers[1].bias" => set_value_opt(
            sam.mask_decoder.output_hypernetworks_mlps[1].layers[1]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].self_attn.v_proj.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[1]
                .self_attn
                .v_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].norm1.gamma" => set_value(
            sam.image_encoder.blocks[11].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[0].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[0].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[5].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[5].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].norm4.gamma" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .norm4
                .gamma
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[0].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].mlp.lin1.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .mlp
                .lin1
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[1].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[1].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[3].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[5].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[5].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].norm2.beta" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .norm2
                .beta
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[9].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_image_to_token.v_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_image_to_token
                    .v_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.transformer.layers[1].cross_attn_token_to_image.q_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .cross_attn_token_to_image
                .q_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[6].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[9].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[4].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].norm2.beta" => set_value(
            sam.image_encoder.blocks[0].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[6].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[11].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].cross_attn_image_to_token.out_proj.weight" => {
            set_value(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_image_to_token
                    .out_proj
                    .weight
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.transformer.final_attn_token_to_image.v_proj.weight" => set_value(
            sam.mask_decoder
                .transformer
                .final_attn_token_to_image
                .v_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[0].layers[0].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[0].layers[0]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].norm1.gamma" => set_value(
            sam.image_encoder.blocks[10].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[1].layers[2].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[1].layers[2]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[11].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.iou_prediction_head.layers[0].bias" => set_value_opt(
            sam.mask_decoder.iou_prediction_head.layers[0]
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].norm1.beta" => set_value(
            sam.image_encoder.blocks[4].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].norm2.gamma" => set_value(
            sam.image_encoder.blocks[7].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[4].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[4].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[8].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.patch_embed.proj.weight" => set_value(
            sam.image_encoder.patch_embed.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_image_to_token.q_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_image_to_token
                    .q_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.output_upscaling0.weight" => set_value(
            sam.mask_decoder.output_upscaling0.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[9].norm1.beta" => set_value(
            sam.image_encoder.blocks[9].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[0].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[7].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[11].norm1.beta" => set_value(
            sam.image_encoder.blocks[11].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_token_to_image.out_proj.weight" => {
            set_value(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_token_to_image
                    .out_proj
                    .weight
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.neck3.bias" => {
            set_value(sam.image_encoder.neck3.bias.borrow_mut(), value, key)
        }
        "mask_decoder.transformer.layers[0].cross_attn_image_to_token.k_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_image_to_token
                    .k_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.transformer.layers[1].cross_attn_token_to_image.out_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[1]
                    .cross_attn_token_to_image
                    .out_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "mask_decoder.transformer.layers[0].cross_attn_image_to_token.v_proj.weight" => set_value(
            sam.mask_decoder.transformer.layers[0]
                .cross_attn_image_to_token
                .v_proj
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[6].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[6].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].norm1.beta" => set_value(
            sam.mask_decoder.transformer.layers[1]
                .norm1
                .beta
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[7].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[7].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[1].self_attn.k_proj.bias" => set_value_opt(
            sam.mask_decoder.transformer.layers[1]
                .self_attn
                .k_proj
                .bias
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[10].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[2].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[2].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.output_hypernetworks_mlps[0].layers[2].weight" => set_value(
            sam.mask_decoder.output_hypernetworks_mlps[0].layers[2]
                .weight
                .borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[8].norm2.beta" => set_value(
            sam.image_encoder.blocks[8].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[10].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[10].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[0].norm1.beta" => set_value(
            sam.image_encoder.blocks[0].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[3].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[3].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "prompt_encoder.mask_downscaling4.bias" => set_value(
            sam.prompt_encoder.mask_downscaling4.bias.borrow_mut(),
            value,
            key,
        ),
        "mask_decoder.transformer.layers[0].cross_attn_token_to_image.out_proj.bias" => {
            set_value_opt(
                sam.mask_decoder.transformer.layers[0]
                    .cross_attn_token_to_image
                    .out_proj
                    .bias
                    .borrow_mut(),
                value,
                key,
            )
        }
        "image_encoder.blocks[14].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[14].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[29].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[15].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[16].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].norm1.gamma" => set_value(
            sam.image_encoder.blocks[12].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[22].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[30].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[27].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[29].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[13].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[31].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].norm1.gamma" => set_value(
            sam.image_encoder.blocks[21].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[31].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[13].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[12].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[15].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].norm1.beta" => set_value(
            sam.image_encoder.blocks[19].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[28].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].norm1.beta" => set_value(
            sam.image_encoder.blocks[24].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[24].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[22].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[30].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].norm2.gamma" => set_value(
            sam.image_encoder.blocks[17].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[18].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[26].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[25].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[21].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[22].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[27].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].norm2.gamma" => set_value(
            sam.image_encoder.blocks[15].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[28].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].norm2.gamma" => set_value(
            sam.image_encoder.blocks[24].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[26].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[16].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[17].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].norm2.beta" => set_value(
            sam.image_encoder.blocks[14].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[28].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[16].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].norm1.gamma" => set_value(
            sam.image_encoder.blocks[14].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[22].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[26].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[14].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[13].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[14].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].norm2.gamma" => set_value(
            sam.image_encoder.blocks[16].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[26].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[31].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].norm2.gamma" => set_value(
            sam.image_encoder.blocks[28].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[23].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[24].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[30].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].norm2.beta" => set_value(
            sam.image_encoder.blocks[25].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[18].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[31].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[22].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[30].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[12].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[15].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[12].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[14].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[17].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].norm2.gamma" => set_value(
            sam.image_encoder.blocks[19].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[19].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[20].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].norm1.beta" => set_value(
            sam.image_encoder.blocks[25].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[21].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[17].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[28].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[30].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].norm1.beta" => set_value(
            sam.image_encoder.blocks[21].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[27].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].norm2.gamma" => set_value(
            sam.image_encoder.blocks[22].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[28].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].norm2.gamma" => set_value(
            sam.image_encoder.blocks[30].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[17].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[15].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[19].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[21].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[21].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[22].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].norm2.gamma" => set_value(
            sam.image_encoder.blocks[31].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].norm1.gamma" => set_value(
            sam.image_encoder.blocks[23].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].norm2.gamma" => set_value(
            sam.image_encoder.blocks[23].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[16].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[25].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[29].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[24].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].norm2.gamma" => set_value(
            sam.image_encoder.blocks[21].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].norm2.beta" => set_value(
            sam.image_encoder.blocks[15].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[25].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[13].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].norm2.beta" => set_value(
            sam.image_encoder.blocks[31].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].norm2.gamma" => set_value(
            sam.image_encoder.blocks[26].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].norm1.gamma" => set_value(
            sam.image_encoder.blocks[30].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[24].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[29].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].norm1.gamma" => set_value(
            sam.image_encoder.blocks[24].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[23].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[24].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[15].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[22].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[30].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[31].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].norm2.beta" => set_value(
            sam.image_encoder.blocks[17].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[23].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[27].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[26].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].norm2.beta" => set_value(
            sam.image_encoder.blocks[24].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[25].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[24].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[23].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[21].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].norm2.beta" => set_value(
            sam.image_encoder.blocks[23].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[13].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[14].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[22].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[23].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[29].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[18].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[16].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].norm1.beta" => set_value(
            sam.image_encoder.blocks[28].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[14].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[27].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].norm2.beta" => set_value(
            sam.image_encoder.blocks[28].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].norm1.gamma" => set_value(
            sam.image_encoder.blocks[19].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[25].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[12].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[20].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[18].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[20].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[26].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[17].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[20].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[14].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[21].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].norm1.beta" => set_value(
            sam.image_encoder.blocks[15].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[18].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[27].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].norm2.beta" => set_value(
            sam.image_encoder.blocks[30].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[13].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].norm1.beta" => set_value(
            sam.image_encoder.blocks[20].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[15].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].norm1.gamma" => set_value(
            sam.image_encoder.blocks[25].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[30].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].norm1.gamma" => set_value(
            sam.image_encoder.blocks[31].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[12].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[18].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[15].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[19].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[29].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].norm2.beta" => set_value(
            sam.image_encoder.blocks[26].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[25].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[30].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[12].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[13].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[27].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[30].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].norm1.gamma" => set_value(
            sam.image_encoder.blocks[17].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[20].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[16].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[24].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].norm1.beta" => set_value(
            sam.image_encoder.blocks[16].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].norm2.gamma" => set_value(
            sam.image_encoder.blocks[13].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[19].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[19].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].norm2.beta" => set_value(
            sam.image_encoder.blocks[16].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[16].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].norm1.gamma" => set_value(
            sam.image_encoder.blocks[16].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].norm2.beta" => set_value(
            sam.image_encoder.blocks[22].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].norm1.beta" => set_value(
            sam.image_encoder.blocks[18].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].norm1.beta" => set_value(
            sam.image_encoder.blocks[23].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[14].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[29].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[24].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[13].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[28].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[12].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].norm1.gamma" => set_value(
            sam.image_encoder.blocks[18].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].norm2.gamma" => set_value(
            sam.image_encoder.blocks[25].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[12].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].norm2.beta" => set_value(
            sam.image_encoder.blocks[21].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[20].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].norm2.beta" => set_value(
            sam.image_encoder.blocks[27].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[23].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].norm1.beta" => set_value(
            sam.image_encoder.blocks[22].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[19].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[28].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[13].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[17].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[18].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].norm2.gamma" => set_value(
            sam.image_encoder.blocks[27].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[25].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].norm2.beta" => set_value(
            sam.image_encoder.blocks[19].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].norm1.beta" => set_value(
            sam.image_encoder.blocks[17].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[19].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[20].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].norm1.gamma" => set_value(
            sam.image_encoder.blocks[29].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].norm1.beta" => set_value(
            sam.image_encoder.blocks[13].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[16].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[28].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[12].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[20].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[12].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[31].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[18].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].norm1.beta" => set_value(
            sam.image_encoder.blocks[12].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[21].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].norm2.gamma" => set_value(
            sam.image_encoder.blocks[18].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].norm1.gamma" => set_value(
            sam.image_encoder.blocks[27].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[30].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].norm2.gamma" => set_value(
            sam.image_encoder.blocks[12].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[24].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[29].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[16].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[21].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].norm1.beta" => set_value(
            sam.image_encoder.blocks[14].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[15].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[20].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[14].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[16].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[16].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[31].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[28].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[14].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[17].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].norm2.beta" => set_value(
            sam.image_encoder.blocks[18].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].norm1.gamma" => set_value(
            sam.image_encoder.blocks[13].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[13].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].norm2.gamma" => set_value(
            sam.image_encoder.blocks[20].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[12].norm2.beta" => set_value(
            sam.image_encoder.blocks[12].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[20].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].norm1.beta" => set_value(
            sam.image_encoder.blocks[26].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[19].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[17].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[18].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[27].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[17].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[30].norm1.beta" => set_value(
            sam.image_encoder.blocks[30].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[27].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[13].norm2.beta" => set_value(
            sam.image_encoder.blocks[13].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[25].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[22].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[19].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].norm2.beta" => set_value(
            sam.image_encoder.blocks[20].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[22].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].mlp.lin1.weight" => set_value(
            sam.image_encoder.blocks[27].mlp.lin1.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[17].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[17].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[20].norm1.gamma" => set_value(
            sam.image_encoder.blocks[20].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[25].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].norm1.gamma" => set_value(
            sam.image_encoder.blocks[26].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[26].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].attn.proj.bias" => set_value_opt(
            sam.image_encoder.blocks[15].attn.proj.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].norm1.gamma" => set_value(
            sam.image_encoder.blocks[15].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].norm1.beta" => set_value(
            sam.image_encoder.blocks[31].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[31].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[31].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[26].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[15].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[15].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[26].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[26].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[26].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[22].norm1.gamma" => set_value(
            sam.image_encoder.blocks[22].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[23].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[23].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].mlp.lin2.weight" => set_value(
            sam.image_encoder.blocks[23].mlp.lin2.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[24].attn.rel_pos_h" => set_value_opt(
            sam.image_encoder.blocks[24].attn.rel_pos_h.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].norm2.gamma" => set_value(
            sam.image_encoder.blocks[29].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[31].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[31].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[29].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].norm1.gamma" => set_value(
            sam.image_encoder.blocks[28].norm1.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[28].attn.rel_pos_w" => set_value_opt(
            sam.image_encoder.blocks[28].attn.rel_pos_w.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].attn.proj.weight" => set_value(
            sam.image_encoder.blocks[21].attn.proj.weight.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].norm2.beta" => set_value(
            sam.image_encoder.blocks[29].norm2.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[14].norm2.gamma" => set_value(
            sam.image_encoder.blocks[14].norm2.gamma.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[27].norm1.beta" => set_value(
            sam.image_encoder.blocks[27].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].norm1.beta" => set_value(
            sam.image_encoder.blocks[29].norm1.beta.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[29].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[29].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[21].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[21].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[19].attn.qkv.bias" => set_value_opt(
            sam.image_encoder.blocks[19].attn.qkv.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[23].mlp.lin1.bias" => set_value_opt(
            sam.image_encoder.blocks[23].mlp.lin1.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[25].mlp.lin2.bias" => set_value_opt(
            sam.image_encoder.blocks[25].mlp.lin2.bias.borrow_mut(),
            value,
            key,
        ),
        "image_encoder.blocks[18].attn.qkv.weight" => set_value(
            sam.image_encoder.blocks[18].attn.qkv.weight.borrow_mut(),
            value,
            key,
        ),
        // _ => panic!("key not found: {}", key),
        _ => _print_match_key(&key),
    }
}
