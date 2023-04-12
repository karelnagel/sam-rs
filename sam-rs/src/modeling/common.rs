use tch::{
    nn::{self, Module},
    Kind, Tensor,
};

use crate::modeling::mask_decoder::Activation;

pub struct MLPBlock {
    lin1: nn::Linear,
    lin2: nn::Linear,
    act: Activation,
}
impl MLPBlock {
    pub fn new(vs: &nn::Path, embedding_dim: i64, mlp_dim: i64, act: Option<Activation>) -> Self {
        let act = act.unwrap_or(Activation::GELU);
        let lin1 = nn::linear(vs, embedding_dim, mlp_dim, Default::default());
        let lin2 = nn::linear(vs, mlp_dim, embedding_dim, Default::default());
        Self { lin1, lin2, act }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = self.lin1.forward(x);
        x = match self.act {
            Activation::GELU => x.gelu("none"),
            Activation::ReLU => x.relu(),
        };
        self.lin2.forward(&x)
    }
}

/// From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
///  Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
pub struct LayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}
impl LayerNorm2d {
    pub fn new(vs: &nn::Path, num_channels: i64, eps: Option<f64>) -> Self {
        let eps = eps.unwrap_or(1e-6);
        let weight = vs.ones("weight", &[num_channels]);
        let bias = vs.zeros("bias", &[num_channels]);
        Self { weight, bias, eps }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let arr: Option<&[i64]> = Some(&[1, 2, 3]);
        let u = x.mean_dim(arr, true, Kind::Float);
        let s = (x.copy() - u.copy())
            .pow_tensor_scalar(2)
            .mean_dim(arr, true, Kind::Float);
        let x = (x.copy() - u.copy()) / (&s + self.eps).sqrt();
        let x =
            &self.weight.unsqueeze(-1).unsqueeze(-1) * x + &self.bias.unsqueeze(-1).unsqueeze(-1);
        x
    }
}
