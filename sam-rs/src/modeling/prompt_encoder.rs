use tch::Tensor;

pub struct PromptEncoder {}

impl PromptEncoder {
    pub fn encode(
        &self,
        point: Option<(Tensor, Tensor)>,
        boxes: Option<Tensor>,
        mask_input: Option<Tensor>,
    ) -> (Tensor, Tensor) {
        // Todo
        (
            Tensor::zeros(&[1, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu)),
            Tensor::zeros(&[1, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu)),
        )
    }
    pub fn get_dense_pe(&self) -> u32 {
        2
    }
}
