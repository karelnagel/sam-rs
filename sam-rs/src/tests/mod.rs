pub mod helpers;
pub mod mocks;

#[cfg(test)]
mod test {
    use ndarray::ArrayD;
    use onnxruntime::environment::Environment;
    use onnxruntime::session::Session;
    use onnxruntime::tensor::OrtOwnedTensor;
    use onnxruntime::GraphOptimizationLevel;
    use opencv::prelude::{Mat, MatTraitConstManual};
    use opencv::{imgcodecs, imgproc};
    extern crate ndarray;
    extern crate opencv;
    use opencv::core::{self};
    use tch::{Device, Kind, Tensor};

    use crate::build_sam::build_sam_vit_h;
    use crate::sam_predictor::{ImageFormat, SamPredictor, Size};

    fn load_image(image_path: &str) -> (Tensor, Size) {
        let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR).unwrap();
        let mut rgb_image = core::Mat::default();
        imgproc::cvt_color(&image, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0).unwrap();
        let arr: ndarray::ArrayView3<u8> = rgb_image.try_as_array();
        let image = Tensor::try_from(arr).unwrap();
        let original_size = Size(image.size()[0], image.size()[1]);
        (image, original_size)
    }

    fn get_ort_env() -> Environment {
        Environment::builder().build().unwrap()
    }
    fn get_ort_session<'a>(pth: &'a str, env: &'a Environment) -> Session<'a> {
        let session: Session = env
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_from_file(pth)
            .unwrap();
        session
    }

    #[test]
    fn test_onnx_model_example() {
        let image_path = "../images/dog.jpg";
        let onnx_model_path = "sam.onnx";
        let sam = build_sam_vit_h(None);
        let mut predictor = SamPredictor::new(sam);

        // Loading image
        let (image, original_size) = load_image(image_path);
        println!("Image shape: {:?}", image.size());
        println!("Original size: {:?}", original_size);
        // Loading model
        let env = get_ort_env();
        let mut ort_session = get_ort_session(onnx_model_path, &env);

        // Setting image
        predictor.set_image(&image, ImageFormat::RGB);
        let image_embedding = predictor.get_image_embedding();
        println!("Image embedding shape: {:?}", image_embedding.size());

        //Example inputs
        let input_point = Tensor::of_slice(&[500, 375]).reshape(&[1, 2]);
        let input_label = Tensor::of_slice(&[1]);

        let onnx_coord = Tensor::cat(
            &[
                input_point,
                Tensor::zeros(&[1, 2], (Kind::Double, Device::Cpu)),
            ],
            0,
        )
        .unsqueeze(0);

        let onnx_label = Tensor::cat(&[input_label, Tensor::of_slice(&[-1])], 0)
            .unsqueeze(0)
            .to_kind(Kind::Double);

        let onnx_coord = predictor.transfrom.apply_coords(&onnx_coord, original_size);

        let onnx_mask_input = Tensor::zeros(&[1, 1, 256, 256], (Kind::Double, tch::Device::Cpu));
        let onnx_has_mask_input = Tensor::zeros(&[1], (Kind::Double, tch::Device::Cpu));

        let orig_im_size =
            Tensor::of_slice(&[original_size.0, original_size.1]).to_kind(Kind::Double);

        let ort_input: Vec<ArrayD<f32>> = vec![
            image_embedding.try_into().unwrap(),
            (&onnx_coord).try_into().unwrap(),
            (&onnx_label).try_into().unwrap(),
            (&onnx_mask_input).try_into().unwrap(),
            (&onnx_has_mask_input).try_into().unwrap(),
            (&orig_im_size).try_into().unwrap(),
        ];
        for i in ort_input.iter() {
            println!("Input shape: {:?} ", i.shape(),);
        }

        let res: Vec<OrtOwnedTensor<f32, _>> = ort_session.run(ort_input).unwrap();
        let masks = Tensor::try_from(res[0].clone()).unwrap();

        let masks = masks.gt(predictor.model.mask_threshold);

        println!("Masks shape: {:?}", masks.size());
        println!("Masks: {:?}", masks);
        panic!("Stop here");
    }

    trait AsArray {
        fn try_as_array(&self) -> ndarray::ArrayView3<u8>;
    }
    impl AsArray for Mat {
        fn try_as_array(&self) -> ndarray::ArrayView3<u8> {
            let bytes = self.data_bytes().unwrap();
            let size = self.size().unwrap();
            let a = ndarray::ArrayView3::from_shape(
                (size.height as usize, size.width as usize, 3),
                bytes,
            )
            .unwrap();
            a
        }
    }
}
