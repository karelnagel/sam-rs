use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, module::conv_transpose2d, ops::ConvTransposeOptions, Tensor},
};

pub struct ConvTranspose2dConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    padding_out: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    bias: bool,
}
impl ConvTranspose2dConfig {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: [usize; 2]) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: [1, 1],
            padding: [0, 0],
            padding_out: [0, 0],
            dilation: [1, 1],
            groups: 1,
            bias: true,
        }
    }
    pub fn set_stride(&mut self, stride: [usize; 2]) -> &mut Self {
        self.stride = stride;
        self
    }
    pub fn set_padding(&mut self, padding: [usize; 2]) -> &mut Self {
        self.padding = padding;
        self
    }
    pub fn set_padding_out(&mut self, padding_out: [usize; 2]) -> &mut Self {
        self.padding_out = padding_out;
        self
    }
    pub fn set_dilation(&mut self, dilation: [usize; 2]) -> &mut Self {
        self.dilation = dilation;
        self
    }
    pub fn set_groups(&mut self, groups: usize) -> &mut Self {
        self.groups = groups;
        self
    }
    pub fn set_bias(&mut self, bias: bool) -> &mut Self {
        self.bias = bias;
        self
    }

    pub fn init<B: Backend>(&self) -> ConvTranspose2d<B> {
        ConvTranspose2d {
            weight: Tensor::ones([
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            ])
            .into(),
            bias: match self.bias {
                true => Some(Tensor::ones([self.out_channels]).into()),
                false => None,
            },
            stride: self.stride,
            padding: self.padding,
            padding_out: self.padding_out,
            dilation: [1, 1],
            groups: 1,
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvTranspose2d<B: Backend> {
    weight: Param<Tensor<B, 4>>,
    bias: Option<Param<Tensor<B, 1>>>,
    stride: [usize; 2],
    padding: [usize; 2],
    padding_out: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl<B: Backend> ConvTranspose2d<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let weight = self.weight.val();

        let res = conv_transpose2d(
            x,
            weight,
            match &self.bias {
                Some(bias) => Some(bias.val()),
                None => None,
            },
            ConvTransposeOptions::new(
                self.stride,
                self.padding,
                self.padding_out,
                self.dilation,
                self.groups,
            ),
        );
        res
    }
}

#[cfg(test)]
mod test {
    use burn_tch::TchBackend;
    use tch::nn::{ConvTransposeConfig, Init, Module};

    use crate::tests::helpers::{random_slice, random_tensor};

    use super::ConvTranspose2dConfig;
    type Backend = TchBackend<f32>;

    fn random_tch_tensor(shape: &[usize], seed: usize) -> tch::Tensor {
        let slice = random_slice(shape, seed);
        let shape = shape.iter().map(|&x| x as i64).collect::<Vec<_>>();
        tch::Tensor::of_slice(&slice).reshape(shape.as_slice())
    }

    #[test]
    fn test_conv_transpose_2d() {
        // Params
        let i: usize = 3;
        let o: usize = 3;
        let k: usize = 3;
        let stride = 1;

        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let tch_conv = tch::nn::conv_transpose2d(
            &vs.root(),
            i as i64,
            o as i64,
            k as i64,
            ConvTransposeConfig {
                stride: stride as i64,
                bs_init: Init::Const(1.),
                ws_init: Init::Const(1.),
                ..Default::default()
            },
        );
        let burn_conv = ConvTranspose2dConfig::new(i, o, [k, k])
            .set_stride([stride, stride])
            .init::<Backend>();

        let shape: [usize; 4] = [2, i, 8, 8];
        let seed = 1;
        let tch_input = random_tch_tensor(&shape, seed);
        let burn_input = random_tensor(shape, seed);

        let tch_output = tch_conv.forward(&tch_input);
        let burn_output = burn_conv.forward(burn_input);
        assert_eq!(
            tch_output.size(),
            burn_output
                .dims()
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<_>>()
        );
        let tch_vec: Vec<f32> = tch_output.flatten(0, -1).into();
        let burn_vec: Vec<f32> = burn_output.to_data().value;
        assert_eq!(tch_vec, burn_vec)
    }
}
