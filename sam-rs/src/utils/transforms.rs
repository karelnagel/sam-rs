use image::{imageops::FilterType, ImageBuffer};
use tch::Tensor;

use crate::sam_predictor::Size;

/// Resizes images to the longest side 'target_length', as well as provides
///  methods for resizing coordinates and boxes. Provides methods for
///  transforming both numpy array and batched torch tensors.
pub struct ResizeLongestSide {
    target_length: i64,
}
impl ResizeLongestSide {
    pub fn new(target_length: i64) -> Self {
        Self { target_length }
    }
    fn resize(image: &Tensor, target_size: Size) -> Tensor {
        let Size(tar_h, tar_w) = target_size;
        let image_data: Vec<u8> = image.into();
        let (width, height) = (image.size()[1], image.size()[0]);
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width as u32, height as u32, image_data).unwrap();
        let resized_img =
            image::imageops::resize(&img, tar_w as u32, tar_h as u32, FilterType::Lanczos3);
        Tensor::of_data_size(
            resized_img.as_raw().as_slice(),
            &[tar_h, tar_w, 3],
            tch::Kind::Uint8,
        )
    }
    // Expects a numpy array with shape HxWxC in uint8 format.
    pub fn apply_image(&self, image: &Tensor) -> Tensor {
        assert!(
            image.kind() == tch::Kind::Uint8,
            "Image must be uint8 format"
        );
        let target_size =
            self.get_preprocess_shape(image.size()[0], image.size()[1], self.target_length);
        return Self::resize(image, target_size);
    }

    // Expects a numpy array of length 2 in the final dimension. Requires the
    // original image size in (H, W) format.
    pub fn apply_coords(&self, coords: &Tensor, original_size: Size) -> Tensor {
        let Size(old_h, old_w) = original_size;
        let Size(new_h, new_w) = self.get_preprocess_shape(old_h, old_w, self.target_length);
        let coords = coords.copy().to_kind(tch::Kind::Double);
        let coords_0 = coords.narrow(-1, 0, 1) * (new_w as f64 / old_w as f64);
        let coords_1 = &coords.narrow(-1, 1, 1) * (new_h as f64 / old_h as f64);
        Tensor::cat(&[&coords_0, &coords_1], -1)
    }

    // Expects a numpy array shape Bx4. Requires the original image size
    // in (H, W) format.
    pub fn apply_boxes(&self, boxes: &Tensor, original_size: Size) -> Tensor {
        let boxes = self.apply_coords(&boxes.reshape(&[-1, 2, 2]), original_size);
        boxes.reshape(&[-1, 4])
    }
    // Expects batched images with shape BxCxHxW and float format. This
    // transformation may not exactly match apply_image. apply_image is
    // the transformation expected by the model.
    pub fn apply_image_torch(&self, image: &Tensor) -> Tensor {
        //  Expects an image in BCHW format. May not exactly match apply_image.

        let (_, _, h, w) = image.size4().unwrap();
        let target_size = self.get_preprocess_shape(h, w, self.target_length);
        image.upsample_bilinear2d(&[target_size.0, target_size.1], false, None, None)
    }

    // Expects a torch tensor with length 2 in the last dimension. Requires the
    // original image size in (H, W) format.
    pub fn apply_coords_torch(&self, coords: &Tensor, original_size: &Size) -> Tensor {
        let Size(old_h, old_w) = original_size;
        let Size(new_h, new_w) = self.get_preprocess_shape(*old_h, *old_w, self.target_length);
        let coords = coords.copy().to_kind(tch::Kind::Float);

        // Update the first column of coords
        let coords_0 = coords.select(1, 0) * (new_w / old_w);
        coords.select(1, 0).copy_(&coords_0);

        // Update the second column of coords
        let coords_1 = coords.select(1, 1) * (new_h / old_h);
        coords.select(1, 1).copy_(&coords_1);
        coords
    }

    // Expects a torch tensor with shape Bx4. Requires the original image
    // size in (H, W) format.
    pub fn apply_boxes_torch(&self, boxes: &Tensor, original_size: &Size) -> Tensor {
        let boxes = self.apply_coords_torch(&boxes.reshape(&[-1, 2, 2]), original_size);
        boxes.reshape(&[-1, 4])
    }

    // Compute the output size given input size and target long side length.
    pub fn get_preprocess_shape(&self, oldh: i64, oldw: i64, long_side_length: i64) -> Size {
        let scale = long_side_length as f64 / oldh.max(oldw) as f64;
        let newh = (oldh as f64 * scale) + 0.5;
        let neww = (oldw as f64 * scale) + 0.5;
        Size(newh as i64, neww as i64)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        sam_predictor::Size,
        tests::helpers::{random_tensor, TestFile},
    };

    #[test]
    fn test_resize_get_preprocess_shape() {
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.get_preprocess_shape(32, 32, 64);
        let file = TestFile::open("resize_get_preprocess_shape");
        file.compare("output", output)
    }
    #[test]
    fn test_resize_apply_image() {
        let resize = super::ResizeLongestSide::new(64);
        let input = (random_tensor(&[120, 180, 3], 1) * 255).to_kind(tch::Kind::Uint8);
        let output = resize.apply_image(&input);
        let file = TestFile::open("resize_apply_image");
        file.compare("input", input);
        file.compare_only_size("output", output);//has similar output but not exact
    }
    #[test]
    fn test_resize_apply_coords() {
        let resize = super::ResizeLongestSide::new(64);
        let input = random_tensor(&[1, 2, 2], 1);
        let original_size = Size(1200, 1800);
        let output = resize.apply_coords(&input, original_size);
        dbg!(&output.flatten(0, -1));
        let file = TestFile::open("resize_apply_coords");
        file.compare("original_size", original_size);
        file.compare("input", input);
        file.compare("output", output);
    }

    #[test]
    fn test_resize_apply_boxes() {
        let resize = super::ResizeLongestSide::new(64);
        let boxes = random_tensor(&[1, 4], 1);
        let original_size = Size(1200, 1800);
        let output = resize.apply_boxes(&boxes, original_size);
        let file = TestFile::open("resize_apply_boxes");
        file.compare("original_size", original_size);
        file.compare("boxes", boxes);
        file.compare("output", output);
    }

    #[test]
    fn test_resize_apply_image_torch() {
        let resize = super::ResizeLongestSide::new(64);
        let input = random_tensor(&[1, 3, 32, 32], 1);
        let output = resize.apply_image_torch(&input);
        let file = TestFile::open("resize_apply_image_torch");
        file.compare("input", input);
        file.compare("output", output);
    }
    #[test]
    fn test_resize_apply_coords_torch() {
        let resize = super::ResizeLongestSide::new(64);
        let coords = random_tensor(&[32, 32], 1);
        let original_size = Size(32, 32);
        let output = resize.apply_coords_torch(&coords, &original_size);
        let file = TestFile::open("resize_apply_coords_torch");
        file.compare("coords", coords);
        file.compare("original_size", original_size);
        file.compare("output", output);
    }
    #[test]
    fn test_resize_apply_boxes_torch() {
        let resize = super::ResizeLongestSide::new(64);
        let boxes = random_tensor(&[32, 32], 1);
        let original_size = Size(32, 32);
        let output = resize.apply_boxes_torch(&boxes, &original_size);
        let file = TestFile::open("resize_apply_boxes_torch");
        file.compare("boxes", boxes);
        file.compare("original_size", original_size);
        file.compare("output", output);
    }
}
