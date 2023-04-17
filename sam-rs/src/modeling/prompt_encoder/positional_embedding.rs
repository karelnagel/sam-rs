use std::ops::Div;

use tch::Tensor;

use crate::sam_predictor::Size;

/// Positional encoding using random spatial frequencies.
#[derive(Debug)]
pub struct PositionEmbeddingRandom {
    positional_encoding_gaussian_matrix: Tensor,
}
impl PositionEmbeddingRandom {
    pub fn new(num_pos_feats: Option<i64>, scale: Option<f32>) -> Self {
        let num_pos_feats = num_pos_feats.unwrap_or(64);
        let mut scale = scale.unwrap_or(1.0);
        if scale <= 0.0 {
            scale = 1.0;
        }

        Self {
            positional_encoding_gaussian_matrix: scale
                * Tensor::randn(&[2, num_pos_feats], (tch::Kind::Float, tch::Device::Cpu)),
        }
    }
    ///Positionally encode points that are normalized to [0,1].
    fn _pe_encoding(&self, coords: &Tensor) -> Tensor {
        let mut coords: Tensor = 2.0 * coords - 1.0;
        coords = coords.matmul(&self.positional_encoding_gaussian_matrix);
        coords = 2.0 * std::f32::consts::PI * coords;
        Tensor::cat(&[&coords.sin(), &coords.cos()], -1)
    }

    /// Generate positional encoding for a grid of the specified size.
    pub fn forward(&self, size: Size) -> Tensor {
        let Size(h, w) = size;
        let device = self.positional_encoding_gaussian_matrix.device();
        let grid = Tensor::ones(&[h, w], (tch::Kind::Float, device));
        let mut y_embed = grid.cumsum(0, tch::Kind::Float) - 0.5;
        let mut x_embed = grid.cumsum(1, tch::Kind::Float) - 0.5;
        y_embed = y_embed / h as f64;
        x_embed = x_embed / w as f64;
        let pe = self._pe_encoding(&Tensor::stack(&[x_embed, y_embed], -1));
        pe.permute(&[2, 0, 1])
    }

    /// Positionally encode points that are not normalized to [0,1].
    pub fn forward_with_coords(&self, coords_input: Tensor, image_size: Size) -> Tensor {
        let coords = coords_input.copy();
        coords
            .narrow(2, 0, 1)
            .copy_(&coords.narrow(2, 0, 1).div(image_size.1 as f64));
        coords
            .narrow(2, 1, 1)
            .copy_(&coords.narrow(2, 1, 1).div(image_size.0 as f64));

        self._pe_encoding(&coords.to_kind(tch::Kind::Float))
    }
}
