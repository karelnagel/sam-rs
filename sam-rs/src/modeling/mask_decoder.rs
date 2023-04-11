use tch::{
    nn::{self, Embedding, Module},
    Tensor,
};
pub enum Activation {
    GELU,
}
pub struct MaskDecoder {
    transformer_dim: i64,
    transformer: Box<dyn Module>,
    num_multimask_outputs: i64,
    iou_token: Embedding,
    num_mask_tokens: i64,
    mask_tokens: Embedding,
    output_upscaling: i64,
    output_hypernetworks_mlps: Vec<MLP>,
    iou_prediction_head: MLP,
}
impl MaskDecoder {
    pub fn new(
        transformer_dim: i64,
        transformer: Box<dyn Module>,
        num_multimask_outputs: i64,
        activation: Activation,
        iou_head_depth: i64,
        iou_head_hidden_dim: i64,
    ) -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let iou_token = nn::embedding(&vs.root(), 1, transformer_dim, Default::default());
        let num_mask_tokens = num_multimask_outputs + 1;
        let mask_tokens = nn::embedding(
            &vs.root(),
            num_mask_tokens,
            transformer_dim,
            Default::default(),
        );
        // Todo it is wrong
        let output_upscaling = nn::seq()
            .add(nn::conv_transpose2d(
                &vs.root(),
                transformer_dim,
                transformer_dim / 4,
                2,
                nn::ConvTransposeConfig {
                    padding: 0,
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add(nn::conv_transpose2d(
                &vs.root(),
                transformer_dim / 4,
                transformer_dim / 8,
                2,
                nn::ConvTransposeConfig {
                    padding: 0,
                    stride: 2,
                    ..Default::default()
                },
            ))
            .len() as i64;

        // Todo maybe wrong too
        let mut output_hypernetworks_mlps = Vec::new();
        for i in 0..num_mask_tokens {
            output_hypernetworks_mlps.push(MLP::new(
                &vs.root(),
                transformer_dim,
                transformer_dim,
                transformer_dim / 8,
                3,
                false, // Sigmoid output (set to true if needed)
            ));
        }
        let iou_prediction_head = MLP::new(
            &vs.root(),
            transformer_dim,
            iou_head_hidden_dim,
            num_mask_tokens,
            iou_head_depth as usize,
            false,
        );
        MaskDecoder {
            transformer_dim,
            transformer,
            num_multimask_outputs,
            num_mask_tokens,
            iou_token,
            mask_tokens,
            output_upscaling,
            output_hypernetworks_mlps,
            iou_prediction_head,
        }
    }
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

// 100% GPT-4 code

#[derive(Debug)]
struct MLP {
    layers: nn::Sequential,
    num_layers: usize,
    sigmoid_output: bool,
}

impl MLP {
    fn new(
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
