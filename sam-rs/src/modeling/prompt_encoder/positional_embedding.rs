use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

use crate::sam_predictor::Size;

/// Positional encoding using random spatial frequencies.
#[derive(Debug, Module)]
pub struct PositionEmbeddingRandom<B: Backend> {
    positional_encoding_gaussian_matrix: Param<Tensor<B, 2>>,
}

impl<B: Backend> PositionEmbeddingRandom<B> {
    pub fn new(num_pos_feats: Option<usize>, scale: Option<f32>) -> Self {
        let num_pos_feats = num_pos_feats.unwrap_or(64);
        let mut scale = scale.unwrap_or(1.0);
        if scale <= 0.0 {
            scale = 1.0;
        }
        let random = Tensor::random([2, num_pos_feats], burn::tensor::Distribution::Standard);

        Self {
            positional_encoding_gaussian_matrix: random.mul_scalar(scale).into(),
        }
    }
    ///Positionally encode points that are normalized to [0,1].
    fn _pe_encoding<const D: usize>(&self, coords: Tensor<B, D>) -> Tensor<B, D> {
        let mut coords = coords.mul_scalar(2.0) - 1.0;
        coords = coords.matmul(self.positional_encoding_gaussian_matrix.val().unsqueeze());
        coords = coords.mul_scalar(2.0 * std::f32::consts::PI);
        Tensor::cat(vec![coords.clone().sin(), coords.cos()], usize::MAX)
    }

    /// Generate positional encoding for a grid of the specified size.
    pub fn forward(&self, size: Size) -> Tensor<B, 3> {
        let Size(h, w) = size;
        let grid = Tensor::ones([h, w]);
        let mut y_embed = grid.cumsum(0) - 0.5;
        let mut x_embed = grid.cumsum(1) - 0.5;
        y_embed = y_embed / h as f64;
        x_embed = x_embed / w as f64;
        let pe: Tensor<B, 3> = self._pe_encoding(Tensor::stack(vec![x_embed, y_embed], usize::MAX));
        pe.permute([2, 0, 1])
    }

    /// Positionally encode points that are not normalized to [0,1].
    pub fn forward_with_coords(&self, coords: Tensor<B, 3>, image_size: Size) -> Tensor<B, 3> {
        let mut coords = coords;
        // coords
        //     .narrow(2, 0, 1) //Todo
        //     .copy_(coords.narrow(2, 0, 1).div_scalar(image_size.1 as f32));
        // coords
        //     .narrow(2, 1, 1)
        //     .copy_(coords.narrow(2, 1, 1).div_scalar(image_size.0 as f32));

        self._pe_encoding(coords)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        sam_predictor::Size,
        tests::helpers::{random_tensor, Test, TestBackend},
    };

    #[test]
    fn test_position_embedding_start() {
        let pos_embedding = super::PositionEmbeddingRandom::<TestBackend>::new(Some(128), None);
        let file = Test::open("position_embedding_random");
        file.compare(
            "gaussian_matrix",
            pos_embedding
                .positional_encoding_gaussian_matrix
                .shape()
                .dims
                .to_vec(),
        );
    }

    #[test]
    fn test_position_embedding_pe_encoding() {
        let mut pos_embedding = super::PositionEmbeddingRandom::<TestBackend>::new(Some(128), None);

        let input = random_tensor([64, 2, 2], 1);
        let output = pos_embedding._pe_encoding(input.clone());
        let file = Test::open("position_embedding_random_pe_encoding");
        file.compare("input", input);
        file.compare("output", output);
    }

    #[test]
    fn test_position_embedding_forward() {
        let mut pos_embedding = super::PositionEmbeddingRandom::<TestBackend>::new(Some(128), None);
        let input = Size(64, 64);
        let output = pos_embedding.forward(input);
        let file = Test::open("position_embedding_random_forward");
        file.compare("input", input);
        file.compare("output", output);
    }

    #[test]
    fn test_position_embedding_forward_with_coords() {
        let mut pos_embedding = super::PositionEmbeddingRandom::<TestBackend>::new(Some(128), None);

        let input = random_tensor([64, 2, 2], 1);
        let image_size = Size(1024, 1024);
        let output = pos_embedding.forward_with_coords(input.clone(), image_size);
        let file = Test::open("position_embedding_random_forward_with_coords");
        file.compare("input", input);
        file.compare("image_size", image_size);
        file.compare("output", output);
    }
}
