use serde::{Deserialize, Serialize};
use tch::{nn::Module, Tensor};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    GELU,
    ReLU,
}

#[derive(Debug, Copy, Clone)]
pub struct Activation {
    act_type: ActivationType,
}

impl Module for Activation {
    fn forward(&self, x: &Tensor) -> Tensor {
        match self.act_type {
            ActivationType::GELU => x.gelu("none"),
            ActivationType::ReLU => x.relu(),
        }
    }
}
impl Activation {
    pub fn new(act_type: ActivationType) -> Self {
        Self { act_type }
    }
}

#[cfg(test)]
mod test {
    use crate::tests::helpers::{random_tensor, TestFile};

    use super::*;

    #[test]
    fn test_activation_gelu() {
        // New
        let act = Activation::new(ActivationType::GELU);
        let input = random_tensor(&[256, 256], 0);
        let output = act.forward(&input);
        let file = TestFile::open("activation_gelu");
        file.compare("input", &input.into());
        file.compare("output", &output.into());
    }

    #[test]
    fn test_activation_relu() {
        // New
        let act = Activation::new(ActivationType::ReLU);
        let input = random_tensor(&[256, 256], 0);
        let output = act.forward(&input);
        let file = TestFile::open("activation_relu");
        file.compare("input", &input.into());
        file.compare("output", &output.into());
    }
}
