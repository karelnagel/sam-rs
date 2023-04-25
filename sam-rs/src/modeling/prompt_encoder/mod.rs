mod positional_embedding;

use self::positional_embedding::PositionEmbeddingRandom;
use super::common::{activation::Activation, layer_norm_2d::LayerNorm2d};
use crate::sam_predictor::Size;
use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Embedding, EmbeddingConfig,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Debug, Module)]
pub struct PromptEncoder<B: Backend> {
    pub embed_dim: usize,
    pub input_image_size: Size,
    image_embedding_size: Size,
    pe_layer: PositionEmbeddingRandom<B>,
    point_embeddings: Vec<Embedding<B>>,
    _num_point_embeddings: usize,
    _mask_input_size: Size,
    no_mask_embed: Embedding<B>,
    not_a_point_embed: Embedding<B>,
    seq1: Conv2d<B>,
    seq2: LayerNorm2d<B>,
    seq3: Activation,
    seq4: Conv2d<B>,
    seq5: LayerNorm2d<B>,
    seq6: Activation,
    seq7: Conv2d<B>,
}

impl<B: Backend> PromptEncoder<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    // Encodes prompts for input to SAM's mask decoder.
    // Arguments:
    //   embed_dim (int): The prompts' embedding dimension
    //   image_embedding_size (tuple(int, int)): The spatial size of the
    //     image embedding, as (H, W).
    //   input_image_size (int): The padded size of the image as input
    //     to the image encoder, as (H, W).
    //   mask_in_chans (int): The number of hidden channels used for
    //     encoding input masks.
    //   activation (nn.Module): The activation to use when encoding
    //     input masks.
    pub fn new(
        embed_dim: usize,
        image_embedding_size: Size,
        input_image_size: Size,
        mask_in_chans: usize,
        activation: Activation,
    ) -> Self {
        let pe_layer = PositionEmbeddingRandom::new(Some(embed_dim / 2), None);
        let num_point_embeddings: usize = 4; // pos/neg point + 2 box corners

        let mut point_embeddings = vec![];
        for _ in 0..num_point_embeddings {
            let point_embedding = EmbeddingConfig::new(1, embed_dim).init();
            point_embeddings.push(point_embedding);
        }
        let not_a_point_embed = EmbeddingConfig::new(1, embed_dim).init();
        let mask_input_size = Size(4 * image_embedding_size.0, 4 * image_embedding_size.1);
        let seq1 = Conv2dConfig::new([1, mask_in_chans / 3], [2, 2]).init();
        let seq2 = LayerNorm2d::new(mask_in_chans / 4, None);
        let seq3 = activation;
        let seq4 = Conv2dConfig::new([mask_in_chans / 4, mask_in_chans], [2, 2]).init();
        let seq5 = LayerNorm2d::new(mask_in_chans, None);
        let seq6 = activation;
        let seq7 = Conv2dConfig::new([mask_in_chans, embed_dim], [1, 1]).init();
        let no_mask_embed = EmbeddingConfig::new(1, embed_dim).init();
        Self {
            embed_dim,
            input_image_size,
            image_embedding_size,
            pe_layer: pe_layer.into(),
            point_embeddings: point_embeddings.into(),
            _num_point_embeddings: num_point_embeddings,
            _mask_input_size: mask_input_size,
            no_mask_embed: no_mask_embed.into(),
            not_a_point_embed: not_a_point_embed.into(),
            seq1,
            seq2,
            seq3,
            seq4,
            seq5,
            seq6,
            seq7,
        }
    }

    // Returns the positional encoding used to encode point prompts,
    // applied to a dense set of points the shape of the image encoding.

    // Returns:
    //   torch.Tensor: Positional encoding with shape
    //     1x(embed_dim)x(embedding_h)x(embedding_w)
    pub fn get_dense_pe(&self) -> Tensor<B, 4> {
        return self.pe_layer.forward(self.image_embedding_size).unsqueeze();
    }

    /// Embeds point prompts.
    fn _embed_points(&self, points: Tensor<B, 3>, labels: Tensor<B, 2>, pad: bool) -> Tensor<B, 3> {
        let mut points = points + 0.5; // Shift to center of pixel
        let mut labels = labels;
        if pad {
            let padding_point = Tensor::zeros([points.shape().dims[0], 1, 2]);
            let padding_label = -Tensor::ones([labels.shape().dims[0], 1]);
            points = Tensor::cat(vec![points, padding_point], 1);
            labels = Tensor::cat(vec![labels, padding_label], 1);
        }
        let mut point_embedding = self
            .pe_layer
            .forward_with_coords(points, self.input_image_size);

        let mask_minus_one = labels.clone().equal_elem(-1.);
        let mask_zero = labels.clone().equal_elem(0.);
        let mask_one = labels.equal_elem(1.);

        point_embedding = point_embedding.zeros_like();
        // .where_self(mask_minus_one.clone().unsqueeze(), point_embedding);

        point_embedding = (point_embedding.clone()
            + self
                .not_a_point_embed
                .clone()
                .into_record()
                .weight
                .val()
                .unsqueeze());
        // .where_self(mask_minus_one.unsqueeze(), point_embedding);

        point_embedding = (point_embedding.clone()
            + self.point_embeddings[0]
                .clone()
                .into_record()
                .weight
                .val()
                .unsqueeze());
        // .where_self(mask_zero.unsqueeze(), point_embedding);

        point_embedding = (point_embedding.clone()
            + self.point_embeddings[1]
                .clone()
                .into_record()
                .weight
                .val()
                .unsqueeze());
        // .where_self(mask_one.unsqueeze(), point_embedding);
        point_embedding
    }

    ///Embeds box prompts.
    fn _embed_boxes(&self, boxes: Tensor<B, 3>) -> Tensor<B, 3> {
        let boxes = boxes + 0.5; // Shift to center of pixel
        let coords = boxes.reshape([usize::MAX, 2, 2]);
        let mut corner_embedding = self
            .pe_layer
            .forward_with_coords(coords, self.input_image_size);

        let corner_embedding_0 = corner_embedding.narrow(1, 0, 1);
        let corner_embedding_1 = corner_embedding.narrow(1, 1, 1);

        let updated_corner_embedding_0 = corner_embedding_0.clone()
            + self.point_embeddings[2]
                .clone()
                .into_record()
                .weight
                .val()
                .unsqueeze();
        // .squeeze_dim(0)
        // .expand_as(corner_embedding_0.clone());
        let updated_corner_embedding_1 = corner_embedding_1.clone()
            + self.point_embeddings[3]
                .clone()
                .into_record()
                .weight
                .val()
                .unsqueeze();
        // .squeeze_dim(0)
        // .expand(corner_embedding_1);

        corner_embedding = Tensor::cat(
            vec![updated_corner_embedding_0, updated_corner_embedding_1],
            1,
        );

        corner_embedding
    }

    ///Embeds mask inputs.
    fn _embed_masks(&self, masks: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut masks = masks;
        masks = self.seq1.forward(masks);
        masks = self.seq2.forward(masks);
        masks = self.seq3.forward(masks);
        masks = self.seq4.forward(masks);
        masks = self.seq5.forward(masks);
        masks = self.seq6.forward(masks);
        masks = self.seq7.forward(masks);
        masks
    }

    /// Gets the batch size of the output given the batch size of the input prompts.
    fn _get_batch_size(
        &self,
        points: Option<(Tensor<B, 3>, Tensor<B, 2>)>,
        boxes: Option<Tensor<B, 3>>,
        masks: Option<Tensor<B, 4>>,
    ) -> usize {
        if let Some((point, _)) = points {
            return point.shape().dims[0];
        } else if let Some(boxes) = boxes {
            return boxes.shape().dims[0];
        } else if let Some(masks) = masks {
            return masks.shape().dims[0];
        } else {
            return 1;
        }
    }

    // Embeds different types of prompts, returning both sparse and dense
    // embeddings.
    // Arguments:
    //   points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
    //     and labels to embed.
    //   boxes (torch.Tensor or none): boxes to embed
    //   masks (torch.Tensor or none): masks to embed
    // Returns:
    //   torch.Tensor: sparse embeddings for the points and boxes, with shape
    //     BxNx(embed_dim), where N is determined by the number of input points
    //     and boxes.
    //   torch.Tensor: dense embeddings for the masks, in the shape
    //     Bx(embed_dim)x(embed_H)x(embed_W)
    pub fn forward(
        &self,
        points: Option<(Tensor<B, 3>, Tensor<B, 2>)>,
        boxes: Option<Tensor<B, 3>>,
        masks: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let bs = self._get_batch_size(points.clone(), boxes.clone(), masks.clone());
        let mut sparse_embeddings = Tensor::empty([bs, 0, self.embed_dim]);
        if let Some((coords, labels)) = points {
            let point_embeddings = self._embed_points(coords, labels, boxes.is_none());
            sparse_embeddings = Tensor::cat(vec![sparse_embeddings, point_embeddings], 1);
        }
        if let Some(boxes) = boxes {
            let box_embeddings = self._embed_boxes(boxes);
            sparse_embeddings = Tensor::cat(vec![sparse_embeddings, box_embeddings], 1);
        }
        let dense_embeddings = match masks {
            Some(masks) => self._embed_masks(masks),
            None => self
                .no_mask_embed
                .clone()
                .into_record()
                .weight
                .val()
                .reshape([1, usize::MAX, 1, 1])
                .expand(
                    vec![
                        bs,
                        usize::MAX,
                        self.image_embedding_size.0,
                        self.image_embedding_size.1,
                    ],
                    true,
                ),
        };
        (sparse_embeddings, dense_embeddings)
    }
}

#[cfg(test)]
mod test {

    use crate::{
        modeling::common::activation::Activation,
        sam_predictor::Size,
        tests::helpers::{random_tensor, Test, TestBackend},
    };

    use super::PromptEncoder;
    const MASK_IN_CHANS: usize = 8;
    const EMBED_DIM: usize = 128;

    fn _init() -> PromptEncoder<TestBackend> {
        let act = Activation::GELU;
        PromptEncoder::new(EMBED_DIM, Size(32, 32), Size(512, 512), MASK_IN_CHANS, act)
    }
    #[test]
    fn test_prompt_encoder_new() {
        let prompt_encoder = _init();

        let file = Test::open("prompt_encoder");
        file.compare("embed_dim", prompt_encoder.embed_dim);
        file.compare("input_image_size", prompt_encoder.input_image_size);
        file.compare("image_embedding_size", prompt_encoder.image_embedding_size);
        file.compare("num_point_embeddings", prompt_encoder._num_point_embeddings);
        file.compare("mask_input_size", prompt_encoder._mask_input_size);
    }

    #[test]
    fn test_prompt_encoder_embed_points() {
        let prompt_encoder = _init();

        let points = random_tensor([32, 1, 2], 1);
        let labels = random_tensor([32, 1], 2);
        let output = prompt_encoder._embed_points(points.clone(), labels.clone(), true);
        let file = Test::open("prompt_encoder_embed_points");
        file.compare("points", points);
        file.compare("labels", labels);
        file.compare("output", output);
    }

    #[test]
    fn test_prompt_encoder_embed_boxes() {
        let mut prompt_encoder = _init();

        let boxes = random_tensor([32, 1, 2], 1);
        let output = prompt_encoder._embed_boxes(boxes.clone());
        let file = Test::open("prompt_encoder_embed_boxes");
        file.compare("boxes", boxes);
        file.compare("output", output);
    }

    #[test]
    fn test_prompt_encoder_embed_masks() {
        let mut prompt_encoder = _init();

        let masks = random_tensor([8, 1, 4, 4], 1);
        let output = prompt_encoder._embed_masks(masks.clone());
        let file = Test::open("prompt_encoder_embed_masks");
        file.compare("masks", masks);
        file.compare("output", output);
    }
    #[test]
    fn test_prompt_encoder_forward() {
        let mut prompt_encoder = _init();

        let points = random_tensor([8, 1, 2], 1);
        let labels = random_tensor([8, 1], 2);
        let (sparse, dense) =
            prompt_encoder.forward(Some((points.clone(), labels.clone())), None, None);
        let file = Test::open("prompt_encoder_forward");
        file.compare("points", points);
        file.compare("labels", labels);
        file.compare("sparse", sparse);
        file.compare("dense", dense);
    }
}
