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

    use crate::tests::helpers::{load_module, random_tensor, Test, TestBackend};

    #[test]
    fn test_mlp() {
        let mut mlp = super::MLP::<TestBackend>::new(256, 256, 256, 4, None);
        mlp = load_module("mlp", mlp);

        // Forward
        let input = random_tensor([1, 256], 1);
        let output = mlp.forward(input.clone());
        let file = Test::open("mlp");
        file.equal("input", input);
        file.almost_equal("output", output, 0.001);
    }
}
