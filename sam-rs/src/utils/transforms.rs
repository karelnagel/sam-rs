use ndarray::{Array1, Array2};

use crate::sam_predictor::{Image, Size};

pub struct ResizeLongestSide {
    target_length: i64,
}

impl ResizeLongestSide {
    /// Resizes images to the longest side 'target_length', as well as provides
    /// methods for resizing coordinates and boxes. Provides methods for
    /// transforming both numpy array and batched torch tensors.
    pub fn new(target_length: i64) -> Self {
        Self { target_length }
    }
    pub fn apply_image(&self, image: &Image) -> Image {
        // Todo
        image.clone()
    }
    pub fn apply_coords(&self, coords: Array2<f32>, size: Size) -> Array2<f32> {
        // Todo
        ndarray::Array2::zeros((1, 1))
    }
    pub fn apply_boxes(&self, boxes: Array1<f32>, size: Size) -> Array1<f32> {
        // Todo
        ndarray::Array1::zeros(1)
    }
}
