#[cfg(test)]
mod test {
    extern crate ndarray;
    extern crate opencv;
    use tch::{Kind, Tensor};

    use crate::build_sam::build_sam_vit_h;
    use crate::onnx_helpers::load_image;
    use crate::sam_predictor::{ImageFormat, SamPredictor};
    use crate::tests::helpers::TestFile;
    use crate::tests::mocks::Mock;

    #[ignore]
    #[test]
    fn test_prediction() {
        let file = TestFile::open("prediction");
        let image_path = "../images/dog.jpg";
        // let checkpoint = Some("../sam_vit_h_4b8939.pth");
        let checkpoint = None;
        let mut sam = build_sam_vit_h(checkpoint);
        sam.mock();
        let mut predictor = SamPredictor::new(sam);

        // Loading image
        let (image, _) = load_image(image_path);
        file.compare("image", image.copy());

        // Setting image
        predictor.set_image(&image, ImageFormat::RGB);

        //Example inputs
        let input_point = Tensor::of_slice(&[170, 375])
            .reshape(&[1, 2])
            .to_kind(Kind::Float);
        let input_label = Tensor::of_slice(&[1.0]).to_kind(Kind::Int);
        file.compare("input_point", input_point.copy());
        file.compare("input_label", input_label.copy());

        let (masks, scores, logits) = predictor.predict(
            Some(&input_point),
            Some(&input_label),
            None,
            None,
            true,
            false,
        );
        file.compare("masks", masks.copy());
        file.compare("scores", scores.copy());
        file.compare("logits", logits.copy());
    }
}
