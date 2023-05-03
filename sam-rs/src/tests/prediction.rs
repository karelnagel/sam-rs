#[cfg(test)]
mod test {
    extern crate ndarray;
    extern crate opencv;

    use burn::tensor::Tensor;

    use crate::build_sam::build_sam_vit_h;
    use crate::burn_helpers::TensorSlice;
    use crate::helpers::load_image;
    use crate::sam_predictor::{ImageFormat, SamPredictor};
    use crate::tests::helpers::{Test, TestBackend};

    #[ignore]
    #[test]
    fn test_prediction() {
        let file = Test::open("prediction");
        let image_path = "../images/dog.jpg";
        // let checkpoint = Some("../sam_vit_h_4b8939.pth");
        let checkpoint = None;
        let sam = build_sam_vit_h::<TestBackend>(checkpoint);
        let mut predictor = SamPredictor::new(sam);

        // Loading image
        let (image, _) = load_image(image_path);
        file.compare("image", image.clone());

        // Setting image
        predictor.set_image(image, ImageFormat::RGB);

        //Example inputs
        let input_point = Tensor::of_slice(vec![170, 375], [1, 2]);
        let input_label = Tensor::of_slice(vec![1.], [1]);
        file.compare("input_point", input_point.clone());
        file.compare("input_label", input_label.clone());

        let (masks, scores, logits) = predictor.predict(
            Some(input_point),
            Some(input_label),
            None,
            None,
            true,
            false,
        );
        file.compare("masks", masks);
        file.compare("scores", scores);
        file.compare("logits", logits);
    }
}
