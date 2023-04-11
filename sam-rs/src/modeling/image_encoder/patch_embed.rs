use tch::{nn, Tensor};

#[derive(Debug)]
pub struct PatchEmbed {
    proj: nn::Conv2D,
}

impl PatchEmbed {
    pub fn new(
        kernel_size: (i64, i64),
        stride: (i64, i64),
        padding: (i64, i64),
        in_chans: i64,
        embed_dim: i64,
    ) -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let proj = nn::conv2d(
            &vs.root(),
            in_chans,
            embed_dim,
            kernel_size,
            nn::ConvConfig {
                stride,
                padding,
                ..Default::default()
            },
        );
        Self { proj }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.proj.forward(x);
        x.permute(&[0, 2, 3, 1])
    }
}

impl nn::Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward(xs)
    }
}
