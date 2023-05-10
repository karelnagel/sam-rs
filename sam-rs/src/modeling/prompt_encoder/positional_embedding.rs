use std::f32::consts::PI;

use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

use crate::{burn_helpers::TensorHelpers, sam_predictor::Size};

/// Positional encoding using random spatial frequencies.
#[derive(Debug, Module, Clone)]
pub struct PositionEmbeddingRandom {
    num_pos_feats: usize,
    scale: f32,
}

impl PositionEmbeddingRandom {
    pub fn new(num_pos_feats: Option<usize>, scale: Option<f32>) -> Self {
        let num_pos_feats = num_pos_feats.unwrap_or(64);
        let mut scale = scale.unwrap_or(1.0);

        if scale <= 0.0 {
            scale = 1.0;
        }
        Self {
            num_pos_feats,
            scale,
        }
    }
    fn positional_encoding_gaussian_matrix<B: Backend>(&self) -> Tensor<B, 2> {
        #[cfg(test)]
        return Tensor::ones([2, self.num_pos_feats]).mul_scalar(self.scale);
        #[cfg(not(test))]
        return Tensor::random(
            [2, self.num_pos_feats],
            burn::tensor::Distribution::Standard, // Todo might be wrong
        )
        .mul_scalar(self.scale);
    }
    ///Positionally encode points that are normalized to [0,1].
    fn _pe_encoding<B: Backend>(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut coords = coords.mul_scalar(2.0) - 1.0;
        coords = coords.matmul(self.positional_encoding_gaussian_matrix().unsqueeze());
        coords = coords.mul_scalar(2.0 * PI);
        Tensor::cat(vec![coords.clone().sin(), coords.cos()], 2)
    }

    /// Generate positional encoding for a grid of the specified size.
    pub fn forward<B: Backend>(&self, size: Size) -> Tensor<B, 3> {
        let Size(h, w) = size;
        let grid = Tensor::ones([h, w]);
        let mut y_embed = grid.cumsum(0) - 0.5;
        let mut x_embed = grid.cumsum(1) - 0.5;
        y_embed = y_embed / h as f64;
        x_embed = x_embed / w as f64;
        let pe: Tensor<B, 3> = self._pe_encoding(Tensor::stack(vec![x_embed, y_embed], 2));
        pe.permute([2, 0, 1])
    }

    /// Positionally encode points that are not normalized to [0,1].
    pub fn forward_with_coords<B: Backend>(
        &self,
        coords: Tensor<B, 3>,
        image_size: Size,
    ) -> Tensor<B, 3> {
        let (slice, shape) = coords.to_slice::<f32>();
        let coords = Tensor::of_slice(slice, shape); // Deep copy
        coords
            .narrow(2, 0, 1)
            .copy_(coords.narrow(2, 0, 1).div_scalar(image_size.1 as f64));
        coords
            .narrow(2, 1, 1)
            .copy_(coords.narrow(2, 1, 1).div_scalar(image_size.0 as f64));
        self._pe_encoding(coords)
    }
}

#[cfg(test)]
mod test {
    use pyo3::{PyResult, Python};

    use crate::{
        python::python_data::{random_python_tensor, PythonData},
        sam_predictor::Size,
        tests::helpers::TestBackend,
    };

    #[test]
    fn test_position_embedding_pe_encoding() {
        fn python() -> PyResult<(PythonData<3>, PythonData<3>)> {
            Python::with_gil(|py| {
                let module = py
                    .import("segment_anything.modeling.prompt_encoder")?
                    .getattr("PositionEmbeddingRandom")?;
                let module = module.call1((128,))?;

                let input = random_python_tensor(py, [64, 69, 2]);
                let output = module.call_method1("_pe_encoding", (input,))?;
                Ok((input.into(), output.into()))
            })
        }
        let (input, python) = python().unwrap();
        let pos_embedding = super::PositionEmbeddingRandom::new(Some(128), None);

        let output = pos_embedding._pe_encoding::<TestBackend>(input.into());
        python.almost_equal(output, 0.1);
    }

    #[test]
    fn test_position_embedding_forward() {
        fn python() -> PyResult<PythonData<3>> {
            Python::with_gil(|py| {
                let module = py
                    .import("segment_anything.modeling.prompt_encoder")?
                    .getattr("PositionEmbeddingRandom")?;
                let module = module.call1((128,))?;

                let output = module.call_method1("forward", ((64, 64),))?;
                Ok(output.into())
            })
        }
        let python = python().unwrap();
        let pos_embedding = super::PositionEmbeddingRandom::new(Some(128), None);

        let output = pos_embedding.forward::<TestBackend>(Size(64, 64));
        python.almost_equal(output, 1.);
    }

    #[test]
    fn test_position_embedding_with_coords() {
        fn python() -> PyResult<(PythonData<3>, PythonData<3>)> {
            Python::with_gil(|py| {
                let module = py
                    .import("segment_anything.modeling.prompt_encoder")?
                    .getattr("PositionEmbeddingRandom")?;
                let module = module.call1((128,))?;
                let input = random_python_tensor(py, [64, 2, 2]);
                let output = module
                    .getattr("forward_with_coords")?
                    .call1((input, (1024, 1024)))?;
                Ok((input.into(), output.into()))
            })
        }
        let (input, python) = python().unwrap();
        let pos_embedding = super::PositionEmbeddingRandom::new(Some(128), None);
        let output =
            pos_embedding.forward_with_coords::<TestBackend>(input.into(), Size(1024, 1024));
        python.almost_equal(output, None);
    }
}
