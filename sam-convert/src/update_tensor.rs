use std::borrow::BorrowMut;

use burn::{
    module::Param,
    tensor::{backend::Backend, Tensor},
};
use pyo3::PyAny;
use sam_rs::{python::python_data::pyany_to_tensor, sam::SamRecord};

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
        _ => panic!("key not found: {}", key),
        // _ => _print_match_key(&key),
    }
}
