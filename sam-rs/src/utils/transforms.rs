use burn::tensor::{backend::Backend, Float, Int, Tensor};
use image::{imageops::FilterType, ImageBuffer};

use crate::{
    burn_helpers::{TensorHelpers, ToFloat},
    sam_predictor::Size,
};

/// Resizes images to the longest side 'target_length', as well as provides
///  methods for resizing coordinates and boxes. Provides methods for
///  transforming both numpy array and batched torch tensors.
pub struct ResizeLongestSide {
    target_length: usize,
}
impl ResizeLongestSide {
    pub fn new(target_length: usize) -> Self {
        Self { target_length }
    }
    fn resize<B: Backend>(image: Tensor<B, 3, Int>, target_size: Size) -> Tensor<B, 3, Int> {
        let Size(tar_h, tar_w) = target_size;
        let (image_data, shape) = image.to_slice::<f32>();
        let image_data = image_data.iter().map(|x| *x as u8).collect::<Vec<u8>>();
        let (width, height) = (shape[1], shape[0]);
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width as u32, height as u32, image_data).unwrap();
        let resized_img =
            image::imageops::resize(&img, tar_w as u32, tar_h as u32, FilterType::Lanczos3);
        Tensor::of_slice(resized_img.into_raw(), [tar_h, tar_w, 3])
    }
    // Expects a numpy array with shape HxWxC in uint8 format.
    pub fn apply_image<B: Backend>(&self, image: Tensor<B, 3, Int>) -> Tensor<B, 3, Int> {
        let shape = image.dims();
        let target_size = self.get_preprocess_shape(shape[0], shape[1], self.target_length);
        return Self::resize(image, target_size);
    }

    // Expects a numpy array of length 2 in the final dimension. Requires the
    // original image size in (H, W) format.
    pub fn apply_coords<B: Backend, const D: usize>(
        &self,
        coords: Tensor<B, D, Int>,
        original_size: Size,
    ) -> Tensor<B, D, Float> {
        let Size(old_h, old_w) = original_size;
        let Size(new_h, new_w) = self.get_preprocess_shape(old_h, old_w, self.target_length);
        let coords = coords.clone().to_float();

        let indexes_0 = (0..D)
            .map(|i| match i == D - 1 {
                true => 0..1,
                false => 0..coords.dims()[i],
            })
            .collect::<Vec<_>>();
        let indexes_1 = (0..D)
            .map(|i| match i == D - 1 {
                true => 1..2,
                false => 0..coords.dims()[i],
            })
            .collect::<Vec<_>>();
        let coords_0 = coords.clone().index::<D>(indexes_0.try_into().unwrap())
            * (new_w as f32 / old_w as f32);
        let coords_1 =
            coords.index::<D>(indexes_1.try_into().unwrap()) * (new_h as f32 / old_h as f32);
        Tensor::cat(vec![coords_0, coords_1], D - 1)
    }

    // Expects a numpy array shape Bx4. Requires the original image size
    // in (H, W) format.
    pub fn apply_boxes<B: Backend>(
        &self,
        boxes: Tensor<B, 2, Int>,
        original_size: Size,
    ) -> Tensor<B, 2, Float> {
        let boxes = self.apply_coords(boxes.reshape_max([usize::MAX, 2, 2]), original_size);
        boxes.reshape_max([usize::MAX, 4])
    }
    // Expects batched images with shape BxCxHxW and float format. This
    // transformation may not exactly match apply_image. apply_image is
    // the transformation expected by the model.
    //  Expects an image in BCHW format. May not exactly match apply_image.
    pub fn apply_image_torch<B: Backend>(&self, image: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        let shape = image.dims();
        let (h, w) = (shape[2], shape[3]);
        let target_size = self.get_preprocess_shape(h, w, self.target_length);
        image.upsample_bilinear2d(vec![target_size.0, target_size.1], false, None, None)
    }

    // Expects a torch tensor with length 2 in the last dimension. Requires the
    // original image size in (H, W) format.
    pub fn apply_coords_torch<B: Backend, const D: usize>(
        &self,
        coords: Tensor<B, D, Int>, //Maybe int
        original_size: Size,
    ) -> Tensor<B, D, Float> {
        let Size(old_h, old_w) = original_size;
        let Size(new_h, new_w) = self.get_preprocess_shape(old_h, old_w, self.target_length);
        let mut coords = coords.clone().to_float();

        let first_dim = 0..coords.dims()[0];

        // Update the first column of coords
        coords = coords
            .mul_scalar(new_w as f32 / old_w as f32);

        // Update the second column of coords
        coords = coords
            .index([first_dim, 0..1])
            .mul_scalar(new_h as f32 / old_h as f32);
        coords
    }

    // Expects a torch tensor with shape Bx4. Requires the original image
    // size in (H, W) format.
    pub fn apply_boxes_torch<B: Backend>(
        &self,
        boxes: Tensor<B, 2, Int>,
        original_size: Size,
    ) -> Tensor<B, 2, Float> {
        let boxes = self.apply_coords_torch(boxes.reshape_max([usize::MAX, 2, 2]), original_size);
        boxes.reshape_max([usize::MAX, 4])
    }

    // Compute the output size given input size and target long side length.
    pub fn get_preprocess_shape(&self, oldh: usize, oldw: usize, long_side_length: usize) -> Size {
        let scale = long_side_length as f32 / oldh.max(oldw) as f32;
        let newh = (oldh as f32 * scale) + 0.5;
        let neww = (oldw as f32 * scale) + 0.5;
        Size(newh as usize, neww as usize)
    }
}

#[cfg(test)]
mod test {
    use pyo3::{PyAny, PyResult, Python};

    use crate::{
        python::python_data::{random_python_tensor, random_python_tensor_int, PythonData},
        sam_predictor::Size,
        tests::helpers::TestBackend,
    };

    fn python_module<'a>(py: &'a Python) -> PyResult<&'a PyAny> {
        let module = py
            .import("segment_anything.utils.transforms")?
            .getattr("ResizeLongestSide")?;
        let module = module.call1((64,))?;
        Ok(module)
    }
    #[test]
    fn test_resize_get_preprocess_shape() {
        let python: PyResult<Size> = Python::with_gil(|py| {
            let module = python_module(&py)?;
            let output = module.call_method1("get_preprocess_shape", (32, 32, 64))?;
            Ok(output.try_into()?)
        });
        let python = python.unwrap();

        let resize = super::ResizeLongestSide::new(64);
        let output = resize.get_preprocess_shape(32, 32, 64);
        assert_eq!(python, output);
    }
    #[test]
    fn test_resize_apply_image() {
        let python: PyResult<(PythonData<3, i64>, PythonData<3, i64>)> = Python::with_gil(|py| {
            let module = python_module(&py)?;
            let uint8 = py.import("torch")?.getattr("uint8")?;
            let input = random_python_tensor(py, [120, 180, 3])?
                .call_method1("type", (uint8,))?
                .call_method0("numpy")?;

            let output = module.call_method1("apply_image", (input,))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_image::<TestBackend>(input.into());
        python.almost_equal(output, 5.); // Resizing a little different
    }
    #[test]
    fn test_resize_apply_coords() {
        let original_size = (1200, 1800);
        let python: PyResult<(PythonData<3>, PythonData<3>)> = Python::with_gil(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor_int(py, [1, 2, 2])?
                .getattr("numpy")?
                .call0()?;
            let output = module.call_method1("apply_coords", (input, original_size))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_coords::<TestBackend, 3>(input.into(), original_size.into());
        python.almost_equal(output, None);
    }

    #[test]
    fn test_resize_apply_boxes() {
        let original_size = (1200, 1800);
        let python: PyResult<(PythonData<2>, PythonData<2>)> = Python::with_gil(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor_int(py, [1, 4])?
                .getattr("numpy")?
                .call0()?;
            let output = module.call_method1("apply_boxes", (input, original_size))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_boxes::<TestBackend>(input.into(), original_size.into());
        python.almost_equal(output, None);
    }

    #[test]
    fn test_resize_image_torch() {
        let python: PyResult<(PythonData<4>, PythonData<4>)> = Python::with_gil(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor(py, [1, 3, 32, 32])?;
            let output = module.call_method1("apply_image_torch", (input,))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_image_torch::<TestBackend>(input.into());
        python.almost_equal(output, None);
    }
    #[test]
    fn test_resize_coords_torch() {
        let size = (32, 32);
        let python: PyResult<(PythonData<2>, PythonData<2>)> = Python::with_gil(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor_int(py, [32, 32])?;
            let output = module.call_method1("apply_coords_torch", (input, size))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_coords_torch::<TestBackend, 2>(input.into(), size.into());
        python.almost_equal(output, None);
    }
    #[test]
    fn test_resize_boxes_torch() {
        let size = (32, 32);
        let python: PyResult<(PythonData<2>, PythonData<2>)> = Python::with_gil(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor_int(py, [32, 32])?;
            let output = module.call_method1("apply_boxes_torch", (input, size))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_boxes_torch::<TestBackend>(input.into(), size.into());
        python.almost_equal(output, None);
    }
}
