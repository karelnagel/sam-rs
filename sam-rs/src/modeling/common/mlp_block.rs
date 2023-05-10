use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::activation::Activation;

#[derive(Debug, Module)]
pub struct MLPBlock<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
    act: Activation,
}
impl<B: Backend> MLPBlock<B> {
    pub fn new(embedding_dim: usize, mlp_dim: usize, act: Activation) -> Self {
        let lin1 = LinearConfig::new(embedding_dim, mlp_dim).init();
        let lin2 = LinearConfig::new(mlp_dim, embedding_dim).init();
        Self { lin1, lin2, act }
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.lin2.forward(self.act.forward(self.lin1.forward(x)))
    }
}

#[cfg(test)]
pub mod test {

    use pyo3::{PyResult, Python};

    use crate::{
        modeling::common::activation::Activation,
        python::module_to_file::module_to_file,
        python::python_data::{random_python_tensor, PythonData},
        tests::helpers::{load_module, TestBackend},
    };

    use super::*;

    #[test]
    fn test_mlp_block() {
        fn python() -> PyResult<(PythonData<2>, PythonData<2>)> {
            Python::with_gil(|py| {
                let module = py
                    .import("segment_anything.modeling.common")?
                    .getattr("MLPBlock")?;

                let mlp_block = module.call1((256, 256))?;
                module_to_file("mlp_block", py, &mlp_block)?;
                let input = random_python_tensor(py, [256, 256])?;
                let output = mlp_block.call1((input,))?;
                Ok((input.try_into()?, output.try_into()?))
            })
        }
        let (input, python) = python().unwrap();
        let mut mlp_block = MLPBlock::<TestBackend>::new(256, 256, Activation::GELU);
        mlp_block = load_module("mlp_block", mlp_block);

        // Forward
        let output = mlp_block.forward(input.into());
        python.almost_equal(output, 0.5);
    }
}
