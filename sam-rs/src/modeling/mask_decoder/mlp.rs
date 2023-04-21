use tch::{nn, Tensor};

#[derive(Debug)]
pub struct MLP {
    layers: Vec<nn::Linear>,
    num_layers: usize,
    sigmoid_output: bool,
}

impl MLP {
    pub fn new(
        vs: &nn::Path,
        input_dim: i64,
        hidden_dim: i64,
        output_dim: i64,
        num_layers: usize,
        sigmoid_output: bool,
    ) -> Self {
        let h = vec![hidden_dim; num_layers - 1];
        let mut layers = vec![];

        let n_values = std::iter::once(input_dim).chain(h.clone());
        let k_values = h.into_iter().chain(std::iter::once(output_dim));
        for (n, k) in n_values.zip(k_values) {
            layers.push(nn::linear(vs, n, k, Default::default()));
        }

        Self {
            layers,
            num_layers,
            sigmoid_output,
        }
    }
}
impl nn::Module for MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.copy();

        for (i, layer) in self.layers.iter().enumerate() {
            x = if i < self.num_layers - 1 {
                layer.forward(&x).relu() // Assuming forward and relu functions are defined
            } else {
                layer.forward(&x)
            };
        }
        if self.sigmoid_output {
            x = x.sigmoid();
        }
        x.copy()
    }
}

#[cfg(test)]
mod test {
    use tch::nn::Module;

    use crate::tests::{
        helpers::{random_tensor, TestFile},
        mocks::Mock,
    };

    impl Mock for super::MLP {
        fn mock(&mut self) {
            for layer in self.layers.iter_mut() {
                layer.mock();
            }
        }
    }
    #[test]
    fn test_mlp() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let mut mlp = super::MLP::new(&vs.root(), 256, 256, 256, 4, false);
        let file = TestFile::open("mlp");
        file.compare("num_layers", mlp.num_layers);
        file.compare("sigmoid_output", mlp.sigmoid_output);
        file.compare("layers_len", mlp.layers.len());
        for (i, layer) in mlp.layers.iter().enumerate() {
            file.compare(&format!("layer{}", i), layer.ws.size());
        }

        // Mocking
        mlp.mock();

        // Forward
        let input = random_tensor(&[1, 256], 1);
        let output = mlp.forward(&input);
        let file = TestFile::open("mlp_forward");
        file.compare("input", input);
        file.compare("output", output);
    }
}
