use tch::Tensor;

pub struct MaskDecoder {}
impl MaskDecoder {
    pub fn decode(
        &self,
        features: &Tensor,
        dense_pe: u32,
        sparse_embeddings: Tensor,
        dense_embeddings: Tensor,
        multimask_output: bool,
    ) -> (Tensor, Tensor) {
        // Todo
        (
            Tensor::zeros(&[1, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu)),
            Tensor::zeros(&[1, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu)),
        )
    }
}
