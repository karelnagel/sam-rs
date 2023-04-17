use tch::{
    nn::{self, Module},
    Kind, Tensor,
};

/// From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
///  Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa

#[derive(Debug)]
pub struct LayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}
impl Module for LayerNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let arr: Option<&[i64]> = Some(&[1]);
        let u = x.mean_dim(arr, true, tch::Kind::Float);
        let s = (x.copy() - u.copy())
            .pow_tensor_scalar(2)
            .mean_dim(arr, true, Kind::Float);
        let x = (x.copy() - u.copy()) / (&s + self.eps).sqrt();
        let x = &self.weight.unsqueeze(1).unsqueeze(2) * x + &self.bias.unsqueeze(1).unsqueeze(2);
        x
    }
}
impl LayerNorm2d {
    pub fn new(vs: &nn::Path, num_channels: i64, eps: Option<f64>) -> Self {
        let eps = eps.unwrap_or(1e-6);
        let weight = Tensor::ones(&[num_channels], (Kind::Float, vs.device()));
        let bias = Tensor::zeros(&[num_channels], (Kind::Float, vs.device()));
        Self { weight, bias, eps }
    }
}

#[cfg(test)]
mod test {
    use crate::tests::helpers::{random_tensor, TestFile};

    use super::*;
    use tch::{nn::VarStore, Device};

    #[test]
    fn test_layer_norm_2d() {
        // New
        let vs = VarStore::new(Device::cuda_if_available());
        let layer_norm = LayerNorm2d::new(&vs.root(), 256, Some(0.1));
        let file = TestFile::open("layer_norm_2d");
        file.compare("weight", layer_norm.weight.copy());
        file.compare("bias", layer_norm.bias.copy());
        file.compare("eps", layer_norm.eps);

        // Forward
        let input = random_tensor(&[2, 256, 16, 16], 0);
        let output = layer_norm.forward(&input);
        let file = TestFile::open("layer_norm_2d_forward");
        file.compare("input", input);
        file.compare("output", output);
    }
}
