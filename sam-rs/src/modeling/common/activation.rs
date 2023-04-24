use burn::{
    module::Module,
    tensor::{
        activation::{gelu, relu},
        backend::Backend,
        Tensor,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    GELU,
    ReLU,
}
#[derive(Debug, Clone, Serialize, Module, Copy)]
pub struct Activation {
    act_type: ActivationType,
}
impl Activation {
    pub fn new(act_type: ActivationType) -> Self {
        Self { act_type }
    }
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let res = match self.act_type {
            ActivationType::GELU => gelu(x),
            ActivationType::ReLU => relu(x),
        };
        res
    }
}
