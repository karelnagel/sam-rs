#[cfg(test)]
mod test {
    extern crate ndarray;
    extern crate opencv;

    use burn::tensor::Tensor;

    use crate::build_sam::BuildSam;
    use crate::burn_helpers::TensorSlice;
    use crate::helpers::load_image;
    use crate::sam_predictor::{ImageFormat, SamPredictor};
    use crate::tests::helpers::{Test, TestBackend};

    #[ignore]
    #[test]
    fn test_prediction() {
        let file = Test::open("prediction");
        let image_path = "../images/dog.jpg";
        let checkpoint = Some("../sam-convert/sam_vit_b_01ec64");
        let sam = BuildSam::SamVitB.build::<TestBackend>(checkpoint);
        let mut predictor = SamPredictor::new(sam);

        // Loading image
        let (image, _) = load_image(image_path);
        file.equal("image", image.clone());

        // Setting image
        predictor.set_image(image, ImageFormat::RGB);

        //Example inputs
        let input_point = Tensor::of_slice(vec![170, 375], [1, 2]);
        let input_label = Tensor::of_slice(vec![1], [1]);
        file.equal("input_point", input_point.clone());
        file.equal("input_label", input_label.clone());

        let (masks, scores, logits) =
            predictor.predict(Some(input_point), Some(input_label), None, None, true);
        let (slice, _) = masks.to_slice();
        println!(
            "true: {:?}, total: {:?}",
            slice.iter().filter(|x| **x).count(),
            slice.len()
        );
        file.equal("masks", masks);
        file.equal("scores", scores);
        file.equal("logits", logits);
    }
}
