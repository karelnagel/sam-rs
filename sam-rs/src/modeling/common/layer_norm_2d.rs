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
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let u = x.clone().mean_dim(1);
        let s = (x.clone() - u.clone()).powf(2.0).mean_dim(1);
        let x = (x - u) / (s + self.eps).sqrt();

        let ws: Tensor<B, 4> = self.weight.val().unsqueeze().swap_dims(4 - 1, 1);
        let bias: Tensor<B, 4> = self.bias.val().unsqueeze().swap_dims(4 - 1, 1);
        ws.mul(x).add(bias)
    }
}

#[cfg(test)]
mod test {
    use pyo3::{PyResult, Python};

    use crate::tests::{
        helpers::TestBackend,
        new::{random_python_tensor, PythonData},
    };

    use super::*;

    #[test]
    fn test_layer_norm_2d() {
        fn python() -> PyResult<(PythonData<4>, PythonData<4>)> {
            Python::with_gil(|py| {
                let common_module = py.import("segment_anything.modeling.common")?;
                let layer_norm_2d = common_module.getattr("LayerNorm2d")?;
                let layer_norm = layer_norm_2d.call1((256, 0.1))?;
                let input = random_python_tensor(py, [2, 256, 16, 16]);
                let output = layer_norm.call1((input,))?;
                Ok((input.into(), output.into()))
            })
        }
        let (input, python) = python().unwrap();
        let layer_norm = LayerNorm2d::<TestBackend>::new(256, Some(0.1));
        let output = layer_norm.forward(input.into());
        python.almost_equal(output, 0.01);
    }
}
