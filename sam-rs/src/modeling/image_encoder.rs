use tch::Tensor;

pub struct ImageEncoder {
    pub img_size: i32,
}

impl ImageEncoder {
    pub fn encode(&self, image: &Tensor) -> Tensor {
        // Todo
        Tensor::zeros(&[1, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu))
    }
}
