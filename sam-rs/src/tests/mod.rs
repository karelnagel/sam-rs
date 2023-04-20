pub mod helpers;
pub mod mocks;

#[cfg(test)]
mod test {
    extern crate ndarray;
    extern crate opencv;
    use tch::{Device, Kind, Tensor};

    use crate::build_sam::build_sam_vit_h;
    use crate::onnx_helpers::{get_ort_env, get_ort_session, load_image, Inference, OnnxInput};
    use crate::sam_predictor::{ImageFormat, SamPredictor};

    #[test]
    fn test_onnx_model_example() {
        let image_path = "../images/dog.jpg";
        let onnx_model_path = "sam.onnx";
        let sam = build_sam_vit_h(None);
        let mut predictor = SamPredictor::new(sam);

        // Loading image
        let (image, original_size) = load_image(image_path);

        // Loading model
        let env = get_ort_env();
        let mut ort_session = get_ort_session(onnx_model_path, &env);

        // Setting image
        predictor.set_image(&image, ImageFormat::RGB);
        let image_embedding = predictor.get_image_embedding();

        //Example inputs
        let input_point = Tensor::of_slice(&[500, 375]).reshape(&[1, 2]);
        let input_label = Tensor::of_slice(&[1]);

        let onnx_label = Tensor::cat(&[input_label, Tensor::of_slice(&[-1])], 0)
            .unsqueeze(0)
            .to_kind(Kind::Double);
        let onnx_coord = Tensor::cat(
            &[
                &input_point,
                &Tensor::zeros(&input_point.size(), (Kind::Double, Device::Cpu)),
            ],
            0,
        )
        .unsqueeze(0);
        let onnx_coord = predictor.transfrom.apply_coords(&onnx_coord, original_size);

        let onnx_mask_input = Tensor::zeros(&[1, 1, 256, 256], (Kind::Double, tch::Device::Cpu));
        let onnx_has_mask_input = Tensor::zeros(&[1], (Kind::Double, tch::Device::Cpu));

        let orig_im_size =
            Tensor::of_slice(&[original_size.0, original_size.1]).to_kind(Kind::Double);

        let ort_input = OnnxInput::new(
            image_embedding.copy(),
            onnx_coord,
            onnx_label,
            onnx_mask_input,
            onnx_has_mask_input,
            orig_im_size,
        );

        let (masks, _, _) = ort_session.inference(ort_input);
        let masks = masks.gt(predictor.model.mask_threshold);
    }
}
