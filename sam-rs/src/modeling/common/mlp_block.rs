use tch::{
    nn::{self, Module},
    Tensor,
};

use super::activation::Activation;

#[derive(Debug)]
pub struct MLPBlock {
    pub lin1: nn::Linear,
    pub lin2: nn::Linear,
    act: Activation,
}
impl MLPBlock {
    pub fn new(vs: &nn::Path, embedding_dim: i64, mlp_dim: i64, act: Activation) -> Self {
        let lin1 = nn::linear(vs, embedding_dim, mlp_dim, Default::default());
        let lin2 = nn::linear(vs, mlp_dim, embedding_dim, Default::default());
        Self { lin1, lin2, act }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.lin2.forward(&self.act.forward(&self.lin1.forward(x)))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        modeling::common::activation::ActivationType,
        test_helpers::{random_tensor, TestFile, ToTest},
    };

    use super::*;
    use tch::{nn::VarStore, Device};

    #[test]
    fn test_mlp_block() {
        // New
        let vs = VarStore::new(Device::cuda_if_available());
        let mut mlp_block =
            MLPBlock::new(&vs.root(), 256, 256, Activation::new(ActivationType::GELU));
        let file = TestFile::open("mlp_block");
        mlp_block.lin1.ws = random_tensor(&[256, 256], 1);
        mlp_block.lin2.ws = random_tensor(&[256, 256], 2);
        mlp_block.lin1.bs = Some(random_tensor(&[256], 3));
        mlp_block.lin2.bs = Some(random_tensor(&[256], 4));
        file.compare("lin1", &mlp_block.lin1.ws.to_test());
        file.compare("lin2", &mlp_block.lin2.ws.to_test());

        // Forward
        let input = random_tensor(&[256, 256], 5);
        let output = mlp_block.forward(&input);
        let file = TestFile::open("mlp_block_forward");
        file.compare("input", &input.to_test());
        file.compare("output", &output.to_test());
    }
}
