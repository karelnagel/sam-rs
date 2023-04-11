use crate::modeling::common::MLPBlock;
use crate::modeling::image_encoder::attention::Attention;
use crate::modeling::image_encoder::helpers::window_partition;
use crate::modeling::image_encoder::helpers::window_unpartition;
use tch::nn::ModuleT;
use tch::Tensor;

#[derive(Debug)]
pub struct Block<'a> {
    norm1: tch::nn::LayerNorm,
    attn: Attention, // Attention module needs to be implemented
    norm2: tch::nn::LayerNorm,
    mlp: MLPBlock<'a>, // MLPBlock module needs to be implemented
    window_size: i64,
}

impl Block {
    pub fn new(
        dim: i64,
        num_heads: i64,
        mlp_ratio: f64,
        qkv_bias: bool,
        use_rel_pos: bool,
        rel_pos_zero_init: bool,
        window_size: i64,
        input_size: Option<(i64, i64)>,
    ) -> Self {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let norm1 = tch::nn::LayerNorm::new(&vs.root(), vec![dim], 1e-5, true);
        let attn = Attention::new(
            // Arguments for Attention module
        );
        let norm2 = tch::nn::LayerNorm::new(&vs.root(), vec![dim], 1e-5, true);
        let mlp = MLPBlock::new(
            // Arguments for MLPBlock module
        );

        Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        }
    }
}

impl ModuleT for Block {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut x = xs;
        let shortcut = x.shallow_clone();
        x = &self.norm1.forward_t(x, train);
        if self.window_size > 0 {
            let (h, w) = (x.size()[1], x.size()[2]);
            x = window_partition(x, self.window_size);
        }

        x = &self.attn.forward_t(&x, train);

        if self.window_size > 0 {
            x = &window_unpartition(x, self.window_size, (h, w));
        }

        x += &shortcut;
        x += self.mlp.forward_t(self.norm2.forward_t(x, train), train);
        x
    }
}
