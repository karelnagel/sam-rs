use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

/// From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
///  Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa

#[derive(Debug, Module)]
pub struct LayerNorm2d<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
    eps: f64,
}
impl<B: Backend> LayerNorm2d<B> {
    pub fn new(num_channels: usize, eps: Option<f64>) -> Self {
        let eps = eps.unwrap_or(1e-6);
        let weight = Tensor::ones([num_channels]);
        let bias = Tensor::zeros([num_channels]);
        Self {
            weight: weight.into(),
            bias: bias.into(),
            eps,
        }
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let u = x.clone().mean_dim(1);
        let s = (x.clone() - u.clone()).powf(2.0).mean_dim(1);
        let x = (x - u) / (s + self.eps).sqrt();

        let ws: Tensor<B, D> = self.weight.val().unsqueeze().transpose();
        let bias: Tensor<B, D> = self.bias.val().reshape(x.shape().dims);
        ws.mul(x).add(bias)
    }
}

#[cfg(test)]
mod test {
    use crate::tests::helpers::{random_tensor, Test, TestBackend};

    use super::*;

    #[test]
    fn test_layer_norm_2d() {
        // New
        let layer_norm = LayerNorm2d::<TestBackend>::new(256, Some(0.1));
        let file = Test::open("layer_norm_2d");
        file.compare("weight", layer_norm.weight.val());
        file.compare("bias", layer_norm.bias.val());
        file.compare("eps", layer_norm.eps);

        // Forward
        let input = random_tensor::<TestBackend, 4>([2, 256, 16, 16], 1);
        let output = layer_norm.forward(input.clone());
        let file = Test::open("layer_norm_2d_forward");
        file.compare("input", input);
        file.compare("output", output);
    }
}
