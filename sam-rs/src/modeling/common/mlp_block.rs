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
    use crate::{
        modeling::common::activation::ActivationType,
        tests::helpers::{random_tensor, Test, TestBackend},
    };

    use super::*;

    #[test]
    fn test_mlp_block() {
        // New
        let mlp_block: MLPBlock<TestBackend> =
            MLPBlock::new(256, 256, Activation::new(ActivationType::GELU));
        // let file = Test::open("mlp_block");

        // Forward
        let input = random_tensor([256, 256], 5);
        let output = mlp_block.forward(input.clone());
        let file = Test::open("mlp_block_forward");
        file.compare("input", input);
        file.compare("output", output);
    }
}
