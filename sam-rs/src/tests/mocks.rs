use tch::{nn, Tensor};

use crate::tests::helpers::random_tensor;

pub trait Mock {
    fn mock(&mut self);
}
impl Mock for nn::Linear {
    fn mock(&mut self) {
        self.ws = random_tensor(&self.ws.size(), 1);
        self.bs = Some(random_tensor(&self.bs.as_ref().unwrap().size(), 2));
    }
}
impl Mock for nn::LayerNorm {
    fn mock(&mut self) {
        self.ws = Some(random_tensor(&self.ws.as_ref().unwrap().size(), 1));
        self.bs = Some(random_tensor(&self.bs.as_ref().unwrap().size(), 2));
    }
}

impl Mock for nn::Conv2D {
    fn mock(&mut self) {
        self.ws = random_tensor(&self.ws.size(), 1);
        if let Some(bs) = &mut self.bs {
            self.bs = Some(random_tensor(&bs.size(), 2));
        }
    }
}
impl Mock for nn::Embedding {
    fn mock(&mut self) {
        self.ws = random_tensor(&self.ws.size(), 1);
    }
}

impl Mock for nn::ConvTranspose2D {
    fn mock(&mut self) {
        self.ws = random_tensor(&self.ws.size(), 1);
        self.bs = Some(random_tensor(&self.bs.as_ref().unwrap().size(), 2));
    }
}

impl Mock for Tensor {
    fn mock(&mut self) {
        self.set_data(&random_tensor(&self.size(), 1))
    }
}
