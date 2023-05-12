#[cfg(test)]
mod test {
    extern crate ndarray;
    extern crate opencv;

    use burn::tensor::Tensor;
    use pyo3::types::{PyDict, PyTuple};
    use pyo3::{PyResult, Python};

    use crate::build_sam::SamVersion;
    use crate::burn_helpers::TensorHelpers;
    use crate::helpers::load_image;
    use crate::python::python_data::PythonData;
    use crate::sam_predictor::{ImageFormat, SamPredictor};
    use crate::tests::helpers::{get_python_sam, get_sam, TestBackend};

    #[ignore]
    #[test]
    fn test_prediction() {
        let image_path = "../images/dog.jpg";
        let version = SamVersion::VitB;
        let checkpoint = Some("../sam-convert/sam_vit_b_01ec64");
        let inputs = vec![170, 375];
        let labels = vec![1];

        let python: PyResult<(
            PythonData<3>,
            PythonData<3>,
            PythonData<1>,
            PythonData<3>,
            PythonData<3>,
        )> = Python::with_gil(|py| {
            let cv2 = py.import("cv2")?;

            // Loading image
            let image = cv2.call_method1("imread", (image_path,))?;
            let image = cv2.call_method1("cvtColor", (image, cv2.getattr("COLOR_BGR2RGB")?))?;

            //Setting image
            let sam = get_python_sam(&py, version, checkpoint)?;
            let predictor = py
                .import("segment_anything.predictor")?
                .call_method1("SamPredictor", (sam,))?;

            predictor.call_method1("set_image", (image,))?;
            let np = py.import("numpy")?;
            let input_point = np.call_method1("array", (vec![inputs.clone()],))?;
            let input_label = np
                .call_method1("array", (labels.clone(),))?
                .call_method1("astype", (np.getattr("int64")?,))?;

            //Predicting
            let kwargs = PyDict::new(py);
            kwargs.set_item("point_coords", input_point)?;
            kwargs.set_item("point_labels", input_label)?;
            kwargs.set_item("multimask_output", true)?;
            let output = predictor
                .call_method("predict", (), Some(kwargs))?
                .downcast::<PyTuple>()?;

            let masks = output.get_item(0)?;
            let scores = output.get_item(1)?;
            let logits = output.get_item(2)?;
            let mask_values = output.get_item(3)?;

            Ok((
                image.try_into()?,
                masks.try_into()?,
                scores.try_into()?,
                logits.try_into()?,
                mask_values.try_into()?,
            ))
        });
        let (image, _masks, scores, logits, mask_values) = python.unwrap();
        let sam = get_sam::<TestBackend>(version, checkpoint);
        let mut predictor = SamPredictor::new(sam);

        // Loading image
        let (image2, _) = load_image(image_path);

        // Setting image
        predictor.set_image(image2.clone(), ImageFormat::RGB);

        //Example inputs
        let input_point = Tensor::of_slice(inputs, [1, 2]);
        let input_label = Tensor::of_slice(labels, [1]);

        let (_masks2, scores2, logits2, mask_values2) =
            predictor.predict(Some(input_point), Some(input_label), None, None, true);
        // masks.almost_equal(masks2, None);
        image.almost_equal(image2, 5.);
        scores.almost_equal(scores2, 5.);
        logits.almost_equal(logits2, 5.);
        mask_values.almost_equal(mask_values2, None);
    }
}
