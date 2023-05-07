use burn::{tensor::{
    activation::{gelu, relu},
    backend::Backend,
    Tensor,
}, module::Module};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Module)]
pub enum Activation {
    GELU,
    ReLU,
}
impl Activation {
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let res = match self {
            Activation::GELU => gelu(x),
            Activation::ReLU => relu(x),
        };
        res
    }
}
