use burn::tensor::{backend::Backend, Int, Tensor};
// use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel};
use opencv::{
    core::Vec3b,
    imgcodecs, imgproc,
    prelude::{Mat, MatTraitConst, MatTraitConstManual},
};

use crate::{burn_helpers::TensorHelpers, sam_predictor::Size};

pub fn load_image_slice(image_path: &str) -> (Vec<u8>, Size) {
    let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR).unwrap();
    let mut rgb_image = Mat::default();
    imgproc::cvt_color(&image, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0).unwrap();

    let size = rgb_image.size().unwrap();
    let shape = Size(size.height as usize, size.width as usize);

    let mut slice = Vec::with_capacity(shape.0 * shape.1 * 3);

    for row in 0..shape.0 {
        for col in 0..shape.1 {
            let pixel: Vec3b = *rgb_image.at_2d(row as i32, col as i32).unwrap();
            for value in pixel {
                slice.push(value);
            }
        }
    }

    (slice, shape)
}

pub fn load_image<B: Backend>(image_path: &str) -> Tensor<B, 3, Int> {
    let (slice, shape) = load_image_slice(image_path);
    let image = Tensor::of_slice(slice, [shape.0, shape.1, 3]);
    image
}

#[cfg(test)]
mod test {

    use pyo3::{PyResult, Python};

    use crate::{python::python_data::PythonData, tests::helpers::TestBackend};

    use super::load_image;

    fn load_python_image(file: &str) -> PyResult<PythonData<3>> {
        Python::with_gil(|py| {
            let cv2 = py.import("cv2")?;
            let image = cv2.call_method1("imread", (file,))?;
            let image = cv2.call_method1("cvtColor", (image, cv2.getattr("COLOR_BGR2RGB")?))?;
            Ok(image.try_into()?)
        })
    }
    #[test]
    fn test_image_loading() {
        let file = "../images/truck.jpg";
        let python_image = load_python_image(file).unwrap();
        let image = load_image::<TestBackend>(file);
        python_image.almost_equal(image, None);
    }
}
