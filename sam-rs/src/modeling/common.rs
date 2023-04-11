use tch::nn::{Linear, Module, ModuleT};
use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct MLPBlock<'a> {
    lin1: Linear,
    lin2: Linear,
    act: tch::nn::func::Func<'a>,
}

impl MLPBlock<'_> {
    fn new(embedding_dim: i64, mlp_dim: i64) -> Self {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let lin1 = Linear::new(&vs.root(), embedding_dim, mlp_dim, Default::default());
        let lin2 = Linear::new(&vs.root(), mlp_dim, embedding_dim, Default::default());
        let act = tch::nn::func::gelu;
        Self { lin1, lin2, act }
    }
}

impl ModuleT for MLPBlock<'_> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.lin2.forward_t(
            &self.act.forward_t(&self.lin1.forward_t(xs, train), train),
            train,
        )
    }
}

#[derive(Debug)]
pub struct LayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm2d {
    pub fn new(num_channels: i64, eps: f64) -> Self {
        let weight = Tensor::ones(&[num_channels], (Kind::Float, tch::Device::Cpu));
        let bias = Tensor::zeros(&[num_channels], (Kind::Float, tch::Device::Cpu));
        Self { weight, bias, eps }
    }
}

impl Module for LayerNorm2d {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let u = xs.mean1(&[1], true, Kind::Float);
        let s = (&xs - &u).pow(2).mean1(&[1], true, Kind::Float);
        let x = (&xs - &u) / (&s + self.eps).sqrt();
        &self.weight.unsqueeze(-1).unsqueeze(-1) * x + &self.bias.unsqueeze(-1).unsqueeze(-1)
    }
}
