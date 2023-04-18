use tch::nn;

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
        self.bs = Some(random_tensor(&self.bs.as_ref().unwrap().size(), 2));
    }
}
impl Mock for nn::Embedding {
    fn mock(&mut self) {
        self.ws = random_tensor(&self.ws.size(), 1);
    }
}
