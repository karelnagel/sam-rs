use tch::Tensor;

#[derive(Debug)]
pub struct TwoWayTransformer {}

// Todo should be this: impl Module for TwoWayTransformer {
impl TwoWayTransformer {
    pub fn forward(
        &self,
        _xs: &Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> (Tensor, Tensor) {
        // Todo
        (
            Tensor::zeros(&[1, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu)),
            Tensor::zeros(&[1, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu)),
        )
    }
}
