use burn::tensor::activation::{relu, sigmoid};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Debug, Module)]
pub struct MLP<B: Backend> {
    layers: Vec<Linear<B>>,
    num_layers: usize,
    sigmoid_output: bool,
}

impl<B: Backend> MLP<B> {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        sigmoid_output: Option<bool>,
    ) -> Self {
        let sigmoid_output = sigmoid_output.unwrap_or(false);
        let h = vec![hidden_dim; num_layers - 1];
        let mut layers = vec![];

        let n_values = std::iter::once(input_dim).chain(h.clone());
        let k_values = h.into_iter().chain(std::iter::once(output_dim));
        for (n, k) in n_values.zip(k_values) {
            layers.push(LinearConfig::new(n, k).init());
        }

        Self {
            layers,
            num_layers,
            sigmoid_output,
        }
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = x;

        for (i, layer) in self.layers.iter().enumerate() {
            x = if i < self.num_layers - 1 {
                relu(layer.forward(x))
            } else {
                layer.forward(x)
            };
        }
        if self.sigmoid_output {
            x = sigmoid(x);
        }
        x
    }
}

#[cfg(test)]
mod test {

    use pyo3::{PyResult, Python};

    use crate::{
        python::{
            module_to_file::module_to_file,
            python_data::{random_python_tensor, PythonData},
        },
        tests::helpers::{load_module, TestBackend},
    };

    #[test]
    fn test_mlp() {
        const FILE: &str = "mlp";
        fn python() -> PyResult<(PythonData<2>, PythonData<2>)> {
            Python::with_gil(|py| {
                let module = py
                    .import("segment_anything.modeling.mask_decoder")?
                    .getattr("MLP")?;
                let module = module.call1((256, 256, 256, 4, false))?;
                module_to_file(FILE, py, &module)?;

                let input = random_python_tensor(py, [1, 256])?;
                let output = module.call1((input,))?;
                Ok((input.try_into()?, output.try_into()?))
            })
        }
        let (input, python) = python().unwrap();
        let mut mlp = super::MLP::<TestBackend>::new(256, 256, 256, 4, None);
        mlp = load_module(FILE, mlp);

        // Forward
        let output = mlp.forward(input.into());
        python.almost_equal(output, None);
    }
}
