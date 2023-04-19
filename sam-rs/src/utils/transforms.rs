use ndarray::{Array1, Array2, Array3};
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

    // Expects a numpy array with shape HxWxC in uint8 format.
    pub fn apply_image(&self, image: &Array3<u8>) -> Array3<u8> {
        unimplemented!();
        // let target_size = self.get_preprocess_shape(
        //     image.shape()[0] as i64,
        //     image.shape()[1] as i64,
        //     self.target_length,
        // );
        // ndarray::Array::from_shape_fn(
        //     (target_size.0 as usize, target_size.1 as usize, 3),
        //     |(i, j, k)| image[[i, j, k]],
        // )
    }

    // Expects a numpy array of length 2 in the final dimension. Requires the
    // original image size in (H, W) format.
    pub fn apply_coords(&self, coords: &Array2<f32>, original_size: &Size) -> Array2<f32> {
        unimplemented!()
        // let Size(old_h, old_w) = original_size;
        // let Size(new_h, new_w) = self.get_preprocess_shape(*old_h, *old_w, self.target_length);
        // let mut coords = coords.to_owned();
        // coords
        //     .column_mut(0)
        //     .mapv_inplace(|x| x * (new_w as f32 / *old_w as f32));
        // coords
        //     .column_mut(1)
        //     .mapv_inplace(|x| x * (new_h as f32 / *old_h as f32));
        // coords
    }

    // Expects a numpy array shape Bx4. Requires the original image size
    // in (H, W) format.
    pub fn apply_boxes(&self, boxes: &Array1<f32>, original_size: &Size) -> Array1<f32> {
        unimplemented!()
        // let boxes = boxes.to_owned();
        // let idk: Vec<i32> = vec![-1, 2, 2];

        // // let idk = boxes.reshape(idk);
        // // let boxes = self.apply_coords(idk, original_size);
        // // boxes.reshape(&[-1, 4])
        // boxes
        // boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        // return boxes.reshape(-1, 4)
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
        let mut coords = coords.to_kind(tch::Kind::Float);
        coords = coords.copy();

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
