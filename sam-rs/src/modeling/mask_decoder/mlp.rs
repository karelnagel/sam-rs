use tch::{nn, Tensor};

#[derive(Debug)]
pub struct MLP {
    layers: nn::Sequential,
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
        let mut layers = nn::seq();
        let mut last_dim = input_dim;
        for i in 0..num_layers {
            let next_dim = if i == num_layers - 1 {
                output_dim
            } else {
                hidden_dim
            };
            layers = layers.add(nn::linear(
                vs / format!("layer_{}", i),
                last_dim,
                next_dim,
                Default::default(),
            ));
            last_dim = next_dim;
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
        let intermediate_outputs = self.layers.forward_all(x, None);
        let mut x = intermediate_outputs[0].shallow_clone();
        for i in 1..self.num_layers {
            x = if i < self.num_layers - 1 {
                x.relu()
            } else {
                intermediate_outputs[i].shallow_clone()
            };
        }
        if self.sigmoid_output {
            x.sigmoid()
        } else {
            x
        }
    }
}

#[cfg(test)]
mod test {
    use tch::nn::Module;

    use crate::test_helpers::{random_tensor, TestFile, ToTest};

    #[test]
    fn test_mlp_block() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let mlp = super::MLP::new(&vs.root(), 256, 256, 256, 4, false);
        let file = TestFile::open("mlp");
        file.compare("num_layers", &mlp.num_layers.to_test());
        file.compare("sigmoid_output", &mlp.sigmoid_output.to_test());

        // Forward
        let input = random_tensor(&[1, 256], 1);
        let output = mlp.forward(&input);
        let file = TestFile::open("mlp_forward");
        file.compare("input", &input.to_test());
        file.compare("output", &output.to_test());
    }
}
