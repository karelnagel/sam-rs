
#[cfg(test)]
mod test {
    extern crate ndarray;
    extern crate opencv;
    use tch::{Device, Kind, Tensor};

    use crate::build_sam::build_sam_vit_h;
    use crate::onnx_helpers::{get_ort_env, get_ort_session, load_image, Inference, OnnxInput};
    use crate::sam_predictor::{ImageFormat, SamPredictor};
    use crate::tests::helpers::TestFile;
    use crate::tests::mocks::Mock;


    #[ignore]
    #[test]
    fn test_onnx_model() {
        let file = TestFile::open("onnx_model");
        let image_path = "../images/dog.jpg";
        let onnx_model_path = "../sam.onnx";
        let mut sam = build_sam_vit_h(None);
        sam.mock();
        let mut predictor = SamPredictor::new(sam);

        // Loading image
        let (image, original_size) = load_image(image_path);
        file.compare("image", image.copy());

        // Loading model
        let env = get_ort_env();
        let mut ort_session = get_ort_session(onnx_model_path, &env);

        // Setting image
        predictor.set_image(&image, ImageFormat::RGB);
        let image_embedding = predictor.get_image_embedding();
        file.compare("image_embedding", image_embedding.copy());

        //Example inputs
        let input_point = Tensor::of_slice(&[100.0, 375.0])
            .reshape(&[1, 2])
            .to_kind(Kind::Float);
        let input_label = Tensor::of_slice(&[1.0]).to_kind(Kind::Float);
        file.compare("input_point", input_point.copy());
        file.compare("input_label", input_label.copy());

        // Creating onnx inputs
        let onnx_coord = Tensor::cat(
            &[
                &input_point,
                &Tensor::zeros(&[1, 2], (Kind::Float, Device::Cpu)),
            ],
            0,
        )
        .unsqueeze(0);
        let onnx_coord = predictor.transfrom.apply_coords(&onnx_coord, original_size);
        let onnx_label = Tensor::cat(&[input_label, Tensor::of_slice(&[-1.0])], 0).unsqueeze(0);
        let onnx_mask_input = Tensor::zeros(&[1, 1, 256, 256], (Kind::Float, tch::Device::Cpu));
        let onnx_has_mask_input = Tensor::zeros(&[1], (Kind::Float, tch::Device::Cpu));
        let orig_im_size =
            Tensor::of_slice(&[original_size.0, original_size.1]).to_kind(Kind::Float);
        file.compare("onnx_coord", onnx_coord.copy());
        file.compare("onnx_label", onnx_label.copy());
        file.compare("onnx_mask_input", onnx_mask_input.copy());
        file.compare("onnx_has_mask_input", onnx_has_mask_input.copy());
        file.compare("orig_im_size", orig_im_size.copy());

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
        file.compare("masks", masks.copy());
    }
}
