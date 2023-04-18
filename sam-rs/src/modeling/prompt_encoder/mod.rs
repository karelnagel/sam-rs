mod positional_embedding;

use tch::{
    nn::{self, Module},
    Device, Tensor,
};

use crate::sam_predictor::Size;

use self::positional_embedding::PositionEmbeddingRandom;

use super::common::{activation::Activation, layer_norm_2d::LayerNorm2d};

#[derive(Debug)]
pub struct PromptEncoder {
    embed_dim: i64,
    input_image_size: Size,
    image_embedding_size: Size,
    pe_layer: PositionEmbeddingRandom,
    point_embeddings: Vec<nn::Embedding>,
    _num_point_embeddings: i64,
    _mask_input_size: Size,
    no_mask_embed: nn::Embedding,
    mask_downscaling: nn::Sequential,
    not_a_point_embed: nn::Embedding,
}

impl PromptEncoder {
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
        vs: &nn::Path,
        embed_dim: i64,
        image_embedding_size: Size,
        input_image_size: Size,
        mask_in_chans: i64,
        activation: Activation,
    ) -> Self {
        let pe_layer = PositionEmbeddingRandom::new(Some(embed_dim / 2), None);
        let num_point_embeddings: i64 = 4; // pos/neg point + 2 box corners

        let mut point_embeddings = vec![];
        for _ in 0..num_point_embeddings {
            let point_embedding = nn::embedding(vs, 1, embed_dim, Default::default());
            point_embeddings.push(point_embedding);
        }
        let not_a_point_embed = nn::embedding(vs, 1, embed_dim, Default::default());

        let mask_input_size = Size(4 * image_embedding_size.0, 4 * image_embedding_size.1);
        let mask_downscaling = nn::seq()
            .add(nn::conv2d(
                vs,
                1,
                mask_in_chans / 4,
                2,
                nn::ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add(LayerNorm2d::new(vs, mask_in_chans / 4, None))
            .add(activation)
            .add(nn::conv2d(
                vs,
                mask_in_chans / 4,
                mask_in_chans,
                2,
                nn::ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add(LayerNorm2d::new(vs, mask_in_chans, None))
            .add(activation)
            .add(nn::conv2d(
                vs,
                mask_in_chans,
                embed_dim,
                1,
                Default::default(),
            ));
        let no_mask_embed = nn::embedding(vs, 1, embed_dim, Default::default());
        Self {
            embed_dim,
            input_image_size,
            image_embedding_size,
            pe_layer,
            point_embeddings,
            _num_point_embeddings: num_point_embeddings,
            _mask_input_size: mask_input_size,
            no_mask_embed,
            mask_downscaling,
            not_a_point_embed,
        }
    }

    // Returns the positional encoding used to encode point prompts,
    // applied to a dense set of points the shape of the image encoding.

    // Returns:
    //   torch.Tensor: Positional encoding with shape
    //     1x(embed_dim)x(embedding_h)x(embedding_w)
    pub fn get_dense_pe(&self) -> Tensor {
        return self
            .pe_layer
            .forward(self.image_embedding_size)
            .unsqueeze(0);
    }

    /// Embeds point prompts.
    fn _embed_points(&self, points: &Tensor, labels: &Tensor, pad: bool) -> Tensor {
        let mut points = points + 0.5; // Shift to center of pixel
        let mut labels = labels.copy();
        if pad {
            let padding_point =
                Tensor::zeros(&[points.size()[0], 1, 2], (tch::Kind::Float, Device::Cpu));
            let padding_label =
                -Tensor::ones(&[labels.size()[0], 1], (tch::Kind::Float, Device::Cpu));
            points = Tensor::cat(&[points, padding_point], 1);
            labels = Tensor::cat(&[labels, padding_label], 1);
        }
        let mut point_embedding = self
            .pe_layer
            .forward_with_coords(&points, self.input_image_size);

        // Todo check if this is correct
        // point_embedding = point_embedding.masked_fill_(&labels.eq(-1), 0.0);
        // point_embedding =
        //     point_embedding.masked_scatter_(&labels.eq(-1), &self.not_a_point_embed.ws);
        // point_embedding =
        //     point_embedding.masked_scatter_(&labels.eq(0), &self.point_embeddings[0].ws);
        // point_embedding =
        //     point_embedding.masked_scatter_(&labels.eq(1), &self.point_embeddings[1].ws);
        point_embedding
    }

    ///Embeds box prompts.
    fn _embed_boxes(&self, boxes: &Tensor) -> Tensor {
        let boxes = boxes + 0.5; // Shift to center of pixel
        let coords = boxes.reshape(&[-1, 2, 2]);
        let mut corner_embedding = self
            .pe_layer
            .forward_with_coords(&coords, self.input_image_size);
        // Todo check
        // corner_embedding = corner_embedding
        //     .masked_scatter_(&boxes.eq(0), &self.point_embeddings[2].ws.unsqueeze(0));
        // corner_embedding = corner_embedding
        //     .masked_scatter_(&boxes.eq(1), &self.point_embeddings[3].ws.unsqueeze(0));
        corner_embedding
    }

    ///Embeds mask inputs.
    fn _embed_masks(&self, masks: &Tensor) -> Tensor {
        let masks = self.mask_downscaling.forward(&masks);
        masks
    }

    /// Gets the batch size of the output given the batch size of the input prompts.
    fn _get_batch_size(
        &self,
        points: Option<(&Tensor, &Tensor)>,
        boxes: Option<&Tensor>,
        masks: Option<&Tensor>,
    ) -> i64 {
        if let Some((point, _)) = points {
            return point.size()[0];
        } else if let Some(boxes) = boxes {
            return boxes.size()[0];
        } else if let Some(masks) = masks {
            return masks.size()[0];
        } else {
            return 1;
        }
    }

    fn _get_device(&self) -> Device {
        return self.point_embeddings[0].ws.device();
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
        points: Option<(&Tensor, &Tensor)>,
        boxes: Option<&Tensor>,
        masks: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let bs = self._get_batch_size(points, boxes, masks);
        let mut sparse_embeddings = Tensor::empty(
            &[bs, 0, self.embed_dim],
            (tch::Kind::Float, self._get_device()),
        );

        if let Some((coords, labels)) = points {
            let point_embeddings = self._embed_points(&coords, &labels, boxes.is_none());
            sparse_embeddings = Tensor::cat(&[sparse_embeddings, point_embeddings], 1);
        }
        if let Some(boxes) = boxes {
            let box_embeddings = self._embed_boxes(&boxes);
            sparse_embeddings = Tensor::cat(&[sparse_embeddings, box_embeddings], 1);
        }

        let dense_embeddings = match masks {
            Some(masks) => self._embed_masks(&masks),
            None => self.no_mask_embed.ws.reshape(&[1, -1, 1, 1]).expand(
                &[bs, self.image_embedding_size.0, self.image_embedding_size.1],
                true,
            ),
        };
        (sparse_embeddings, dense_embeddings)
    }
}

#[cfg(test)]
mod test {
    use tch::{nn, Device};

    use crate::{
        modeling::common::{
            activation::{Activation, ActivationType},
            layer_norm_2d::LayerNorm2d,
        },
        sam_predictor::Size,
        tests::{
            helpers::{random_tensor, TestFile},
            mocks::Mock,
        },
    };

    use super::PromptEncoder;
    const MASK_IN_CHANS: i64 = 16;
    const EMBED_DIM: i64 = 256;

    impl Mock for PromptEncoder {
        fn mock(&mut self) {
            self.pe_layer.mock();
            self.no_mask_embed.mock();
            self.not_a_point_embed.mock();
            for item in self.point_embeddings.iter_mut() {
                item.mock();
            }
            let vs = nn::VarStore::new(Device::Cpu);
            let activation = Activation::new(ActivationType::GELU);
            let mut conv1 = nn::conv2d(
                &vs.root(),
                1,
                MASK_IN_CHANS / 4,
                2,
                nn::ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            );
            let mut conv2 = nn::conv2d(
                &vs.root(),
                MASK_IN_CHANS / 4,
                MASK_IN_CHANS,
                2,
                nn::ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            );
            let mut conv3 = nn::conv2d(&vs.root(), MASK_IN_CHANS, EMBED_DIM, 1, Default::default());
            conv1.mock();
            conv2.mock();
            conv3.mock();
            self.mask_downscaling = nn::seq()
                .add(conv1)
                .add(LayerNorm2d::new(&vs.root(), MASK_IN_CHANS / 4, None))
                .add(activation)
                .add(conv2)
                .add(LayerNorm2d::new(&vs.root(), MASK_IN_CHANS, None))
                .add(activation)
                .add(conv3);
        }
    }
    fn _init() -> PromptEncoder {
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let act = Activation::new(ActivationType::GELU);
        let prompt_encoder = PromptEncoder::new(
            &vs.root(),
            EMBED_DIM,
            Size(64, 64),
            Size(1024, 1024),
            MASK_IN_CHANS,
            act,
        );
        prompt_encoder
    }
    #[test]
    fn test_prompt_encoder_new() {
        let prompt_encoder = _init();

        let file = TestFile::open("prompt_encoder");
        file.compare("embed_dim", prompt_encoder.embed_dim);
        file.compare("input_image_size", prompt_encoder.input_image_size);
        file.compare("image_embedding_size", prompt_encoder.image_embedding_size);
        file.compare("num_point_embeddings", prompt_encoder._num_point_embeddings);
        file.compare("mask_input_size", prompt_encoder._mask_input_size);
    }

    #[test]
    fn test_prompt_encoder_embed_points() {
        let mut prompt_encoder = _init();
        prompt_encoder.mock();

        let points = random_tensor(&[64, 1, 2], 1);
        let labels = random_tensor(&[64, 1], 1);
        let output = prompt_encoder._embed_points(&points, &labels, true);
        let file = TestFile::open("prompt_encoder_embed_points");
        file.compare("points", points);
        file.compare("labels", labels);
        file.compare("output", output);
    }

    #[test]
    fn test_prompt_encoder_embed_boxes() {
        let mut prompt_encoder = _init();
        prompt_encoder.mock();

        let boxes = random_tensor(&[64, 1, 2], 1);
        let output = prompt_encoder._embed_boxes(&boxes);
        let file = TestFile::open("prompt_encoder_embed_boxes");
        file.compare("boxes", boxes);
        file.compare("output", output);
    }

    #[ignore]
    #[test]
    fn test_prompt_encoder_embed_masks() {
        let mut prompt_encoder = _init();
        prompt_encoder.mock();

        let masks = random_tensor(&[64, MASK_IN_CHANS, 64, 64], 1);
        let output = prompt_encoder._embed_masks(&masks);
        let file = TestFile::open("prompt_encoder_embed_masks");
        file.compare("masks", masks);
        file.compare("output", output);
    }

    #[test]
    fn test_prompt_encoder_forward() {
        let mut prompt_encoder = _init();
        prompt_encoder.mock();

        let points = random_tensor(&[64, 1, 2], 1);
        let labels = random_tensor(&[64, 1], 1);
        let boxes = random_tensor(&[64, 1, 2], 1);
        let masks = random_tensor(&[64, MASK_IN_CHANS, 64, 64], 1);
        let (sparse, dense) =
            prompt_encoder.forward(Some((&points, &labels)), Some(&boxes), Some(&masks));
        let file = TestFile::open("prompt_encoder_forward");
        file.compare("points", points);
        file.compare("labels", labels);
        file.compare("boxes", boxes);
        file.compare("masks", masks);
        file.compare("sparse", sparse);
        file.compare("dense", dense);
    }
}
