use burn::tensor::{backend::Backend, Tensor};
use image::{imageops::FilterType, ImageBuffer};

use crate::{
    burn_helpers::{TensorHelpers, TensorSlice},
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
    fn resize<B: Backend>(image: Tensor<B, 3>, target_size: Size) -> Tensor<B, 3> {
        let Size(tar_h, tar_w) = target_size;
        let (image_data, shape): (Vec<f32>, [usize; 3]) = image.to_slice();
        let image_data = image_data.iter().map(|x| *x as u8).collect::<Vec<u8>>();
        let (width, height) = (shape[1], shape[0]);
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width as u32, height as u32, image_data).unwrap();
        let resized_img =
            image::imageops::resize(&img, tar_w as u32, tar_h as u32, FilterType::Lanczos3);
        Tensor::of_slice(resized_img.into_raw(), [tar_h, tar_w, 3])
    }
    // Expects a numpy array with shape HxWxC in uint8 format.
    pub fn apply_image<B: Backend>(&self, image: Tensor<B, 3>) -> Tensor<B, 3> {
        // assert!(
        //     image.kind() == tch::Kind::Uint8,
        //     "Image must be uint8 format"
        // );
        let shape = image.dims();
        let target_size = self.get_preprocess_shape(shape[0], shape[1], self.target_length);
        return Self::resize(image, target_size);
    }

    // Expects a numpy array of length 2 in the final dimension. Requires the
    // original image size in (H, W) format.
    pub fn apply_coords<B: Backend>(
        &self,
        coords: Tensor<B, 2>,
        original_size: Size,
    ) -> Tensor<B, 2> {
        let Size(old_h, old_w) = original_size;
        let Size(new_h, new_w) = self.get_preprocess_shape(old_h, old_w, self.target_length);
        let coords = coords.clone();
        let coords_0 = coords.narrow(usize::MAX, 0, 1) * (new_w as f64 / old_w as f64);
        let coords_1 = coords.narrow(usize::MAX, 1, 1) * (new_h as f64 / old_h as f64);
        Tensor::cat(vec![coords_0, coords_1], usize::MAX)
    }

    // Expects a numpy array shape Bx4. Requires the original image size
    // in (H, W) format.
    pub fn apply_boxes<B: Backend>(
        &self,
        boxes: Tensor<B, 2>,
        original_size: Size,
    ) -> Tensor<B, 2> {
        let boxes = self.apply_coords(boxes, original_size);
        boxes.reshape_max([usize::MAX, 4])
    }
    // Expects batched images with shape BxCxHxW and float format. This
    // transformation may not exactly match apply_image. apply_image is
    // the transformation expected by the model.
    //  Expects an image in BCHW format. May not exactly match apply_image.
    pub fn apply_image_torch<B: Backend>(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = image.dims();
        let (h, w) = (shape[2], shape[3]);
        let target_size = self.get_preprocess_shape(h, w, self.target_length);
        image.upsample_bilinear2d(vec![target_size.0, target_size.1], false, None, None)
    }

    // Expects a torch tensor with length 2 in the last dimension. Requires the
    // original image size in (H, W) format.
    pub fn apply_coords_torch<B: Backend>(
        &self,
        coords: Tensor<B, 3>,
        original_size: Size,
    ) -> Tensor<B, 3> {
        let Size(old_h, old_w) = original_size;
        let Size(new_h, new_w) = self.get_preprocess_shape(old_h, old_w, self.target_length);
        let mut coords = coords.clone();

        // Update the first column of coords
        let coords_0 = coords
            .select::<2>(1, 0)
            .mul_scalar(new_w as f32 / old_w as f32);
        coords = coords.select(1, 0);

        // Update the second column of coords
        let coords_1 = coords
            .select::<2>(1, 1)
            .mul_scalar(new_h as f32 / old_h as f32);
        coords = coords.select(1, 1);
        coords
    }

    // Expects a torch tensor with shape Bx4. Requires the original image
    // size in (H, W) format.
    pub fn apply_boxes_torch<B: Backend>(
        &self,
        boxes: Tensor<B, 2>,
        original_size: Size,
    ) -> Tensor<B, 2> {
        let boxes = self.apply_coords_torch(boxes.reshape_max([usize::MAX, 2, 2]), original_size);
        boxes.reshape_max([usize::MAX, 4])
    }

    // Compute the output size given input size and target long side length.
    pub fn get_preprocess_shape(&self, oldh: usize, oldw: usize, long_side_length: usize) -> Size {
        let scale = long_side_length as f64 / oldh.max(oldw) as f64;
        let newh = (oldh as f64 * scale) + 0.5;
        let neww = (oldw as f64 * scale) + 0.5;
        Size(newh as usize, neww as usize)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        sam_predictor::Size,
        tests::helpers::{random_tensor, Test, TestBackend},
    };

    #[test]
    fn test_resize_get_preprocess_shape() {
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.get_preprocess_shape(32, 32, 64);
        let file = Test::open("resize_get_preprocess_shape");
        file.compare("output", output)
    }
    #[test]
    fn test_resize_apply_image() {
        let resize = super::ResizeLongestSide::new(64);
        let input = random_tensor::<TestBackend, 3>([120, 180, 3], 1) * 255;
        let output = resize.apply_image(input.clone());
        let file = Test::open("resize_apply_image");
        file.compare("input", input);
        file.compare("output", output); //has similar output but not exact
    }
    #[test]
    fn test_resize_apply_coords() {
        let resize = super::ResizeLongestSide::new(64);
        let input = random_tensor::<TestBackend, 2>([1, 5], 1);
        let original_size = Size(1200, 1800);
        let output = resize.apply_coords(input.clone(), original_size);
        let file = Test::open("resize_apply_coords");
        file.compare("original_size", original_size);
        file.compare("input", input);
        file.compare("output", output);
    }

    #[test]
    fn test_resize_apply_boxes() {
        let resize = super::ResizeLongestSide::new(64);
        let boxes = random_tensor::<TestBackend, 2>([1, 4], 1);
        let original_size = Size(1200, 1800);
        let output = resize.apply_boxes(boxes.clone(), original_size);
        let file = Test::open("resize_apply_boxes");
        file.compare("original_size", original_size);
        file.compare("boxes", boxes);
        file.compare("output", output);
    }

    #[test]
    fn test_resize_apply_image_torch() {
        let resize = super::ResizeLongestSide::new(64);
        let input = random_tensor::<TestBackend, 4>([1, 3, 32, 32], 1);
        let output = resize.apply_image_torch(input.clone());
        let file = Test::open("resize_apply_image_torch");
        file.compare("input", input);
        file.compare("output", output);
    }
    #[test]
    fn test_resize_apply_coords_torch() {
        let resize = super::ResizeLongestSide::new(64);
        let coords = random_tensor::<TestBackend, 3>([32, 32, 2], 1);
        let original_size = Size(32, 32);
        let output = resize.apply_coords_torch(coords.clone(), original_size);
        let file = Test::open("resize_apply_coords_torch");
        file.compare("coords", coords);
        file.compare("original_size", original_size);
        file.compare("output", output);
    }
    #[test]
    fn test_resize_apply_boxes_torch() {
        let resize = super::ResizeLongestSide::new(64);
        let boxes = random_tensor::<TestBackend, 2>([32, 32], 1);
        let original_size = Size(32, 32);
        let output = resize.apply_boxes_torch(boxes.clone(), original_size);
        let file = Test::open("resize_apply_boxes_torch");
        file.compare("boxes", boxes);
        file.compare("original_size", original_size);
        file.compare("output", output);
    }
}
