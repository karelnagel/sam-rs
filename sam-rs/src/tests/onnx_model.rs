#[cfg(test)]
mod test {
    extern crate ndarray;
    extern crate opencv;

    use burn::tensor::Tensor;

    use crate::build_sam::build_sam_vit_h;
    use crate::onnx_helpers::{get_ort_env, get_ort_session, load_image, Inference, OnnxInput};
    use crate::sam_predictor::{ImageFormat, SamPredictor};
    use crate::tests::helpers::{Test, TestBackend};

    #[ignore]
    #[test]
    fn test_onnx_model() {
        let file = Test::open("onnx_model");
        let image_path = "../images/dog.jpg";
        let onnx_model_path = "../sam.onnx";
        let mut sam = build_sam_vit_h(None);
        let mut predictor = SamPredictor::new(sam);

        // Loading image
        let (image, original_size) = load_image(image_path);
        file.compare("image", image);

        // Loading model
        let env = get_ort_env();
        let mut ort_session = get_ort_session(onnx_model_path, &env);

        // Setting image
        predictor.set_image(image, ImageFormat::RGB);
        let image_embedding = predictor.get_image_embedding();
        file.compare("image_embedding", image_embedding);

        //Example inputs
        let input_point = Tensor::of_slice(vec![100.0, 375.0], [1, 2]);
        let input_label = Tensor::of_slice(vec![1.0], [1]);
        file.compare("input_point", input_point);
        file.compare("input_label", input_label);

        // Creating onnx inputs
        let onnx_coord = Tensor::cat(vec![input_point, Tensor::zeros([1, 2])], 0).unsqueeze();
        let onnx_coord = predictor.transfrom.apply_coords(onnx_coord, original_size);
        let onnx_label = Tensor::cat(vec![input_label, Tensor::of_slice(vec![-1.0], [1])], 0);
        let onnx_mask_input = Tensor::zeros([1, 1, 256, 256]);
        let onnx_has_mask_input = Tensor::zeros([1]);
        let orig_im_size =
            Tensor::of_slice(vec![original_size.0 as f32, original_size.1 as f32], [2]);
        file.compare("onnx_coord", onnx_coord);
        file.compare("onnx_label", onnx_label);
        file.compare("onnx_mask_input", onnx_mask_input);
        file.compare("onnx_has_mask_input", onnx_has_mask_input);
        file.compare("orig_im_size", orig_im_size);

        let ort_input = OnnxInput::<TestBackend>::new(
            image_embedding,
            onnx_coord,
            onnx_label,
            onnx_mask_input,
            onnx_has_mask_input,
            orig_im_size,
        );

        let (masks, scores, logits) = ort_session.inference(ort_input);
        let masks = masks.greater_elem(predictor.model.mask_threshold);
        file.compare("masks", masks);
        file.compare("scores", scores);
        file.compare("logits", logits);
    }
}
