use std::ops::Div;

use tch::{nn, Device, Tensor};

use crate::sam_predictor::Size;

use super::common::activation::Activation;

#[derive(Debug)]
pub struct PromptEncoder {
    embed_dim: i64,
    input_image_size: Size,
    image_embedding_size: Size,
    pe_layer: PositionEmbeddingRandom,
    point_embeddings: Vec<nn::Embedding>,
    num_point_embeddings: i64,
    mask_input_size: Size,
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
        let point_embeddings = (0..num_point_embeddings)
            .map(|_| nn::embedding(vs, 1, embed_dim, Default::default()))
            .collect::<Vec<_>>();
        let not_a_point_embed = nn::embedding(vs, 1, embed_dim, Default::default());
        let mask_input_size = Size(4 * image_embedding_size.0, 4 * image_embedding_size.1);
        let mask_downscaling = nn::seq()
            .add(nn::conv2d(
                vs,
                1,
                mask_in_chans / 4,
                2,
                nn::ConvConfig {
                    padding: 0,
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add(nn::layer_norm(
                vs,
                vec![mask_in_chans / 4],
                Default::default(),
            ))
            .add(activation)
            .add(nn::conv2d(
                vs,
                mask_in_chans / 4,
                mask_in_chans / 8,
                2,
                nn::ConvConfig {
                    padding: 0,
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add(nn::layer_norm(
                vs,
                vec![mask_in_chans / 8],
                Default::default(),
            ))
            .add(activation);
        let no_mask_embed = nn::embedding(vs, 1, embed_dim, Default::default());
        Self {
            embed_dim,
            input_image_size,
            image_embedding_size,
            pe_layer,
            point_embeddings,
            num_point_embeddings,
            mask_input_size,
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
    fn _embed_points(&self, points: Tensor, labels: Tensor, pad: bool) -> Tensor {
        let mut points = points + 0.5; // Shift to center of pixel
        let mut labels = labels;
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
            .forward_with_coords(points, self.input_image_size);

        point_embedding = point_embedding.masked_fill_(&labels.eq(-1), 0.0);
        point_embedding =
            point_embedding.masked_scatter_(&labels.eq(-1), &self.not_a_point_embed.ws);
        point_embedding =
            point_embedding.masked_scatter_(&labels.eq(0), &self.point_embeddings[0].ws);
        point_embedding =
            point_embedding.masked_scatter_(&labels.eq(1), &self.point_embeddings[1].ws);
        point_embedding
    }

    ///Embeds box prompts.
    fn _embed_boxes(&self, boxes: Tensor) -> Tensor {
        let boxes = boxes + 0.5; // Shift to center of pixel
        let coords = boxes.reshape(&[-1, 2, 2]);
        let mut corner_embedding = self
            .pe_layer
            .forward_with_coords(coords, self.input_image_size);
        corner_embedding = corner_embedding
            .masked_scatter_(&boxes.eq(0), &self.point_embeddings[2].ws.unsqueeze(0));
        corner_embedding = corner_embedding
            .masked_scatter_(&boxes.eq(1), &self.point_embeddings[3].ws.unsqueeze(0));
        corner_embedding
    }

    ///Embeds mask inputs.
    fn _embed_masks(&self, masks: Tensor) -> Tensor {
        let masks = self.mask_downscaling.forward_all(&masks, None);
        masks.get(0).unwrap().copy()
    }

    /// Gets the batch size of the output given the batch size of the input prompts.
    fn _get_batch_size(
        &self,
        points: &Option<(Tensor, Tensor)>,
        boxes: &Option<Tensor>,
        masks: &Option<Tensor>,
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
        points: Option<(Tensor, Tensor)>,
        boxes: Option<Tensor>,
        masks: Option<Tensor>,
    ) -> (Tensor, Tensor) {
        let bs = self._get_batch_size(&points, &boxes, &masks);
        let mut sparse_embeddings = Tensor::empty(
            &[bs, 0, self.embed_dim],
            (tch::Kind::Float, self._get_device()),
        );

        if let Some((coords, labels)) = points {
            let point_embeddings = self._embed_points(coords, labels, true);
            sparse_embeddings = Tensor::cat(&[sparse_embeddings, point_embeddings], 1);
        }
        if let Some(boxes) = boxes {
            let box_embeddings = self._embed_boxes(boxes);
            sparse_embeddings = Tensor::cat(&[sparse_embeddings, box_embeddings], 1);
        }

        if let Some(masks) = masks {
            let dense_embeddings = self._embed_masks(masks);
            (sparse_embeddings, dense_embeddings)
        } else {
            let dense_embeddings = self.no_mask_embed.ws.reshape(&[1, -1, 1, 1]).expand(
                &[
                    bs,
                    self.no_mask_embed.ws.size()[0],
                    self.no_mask_embed.ws.size()[1],
                    self.no_mask_embed.ws.size()[2],
                ],
                true,
            );
            (sparse_embeddings, dense_embeddings)
        }
    }
}

/// Positional encoding using random spatial frequencies.
#[derive(Debug)]
struct PositionEmbeddingRandom {
    positional_encoding_gaussian_matrix: Tensor,
}
impl PositionEmbeddingRandom {
    pub fn new(num_pos_feats: Option<i64>, scale: Option<f32>) -> Self {
        let num_pos_feats = num_pos_feats.unwrap_or(64);
        let mut scale = scale.unwrap_or(1.0);
        if scale <= 0.0 {
            scale = 1.0;
        }

        Self {
            positional_encoding_gaussian_matrix: scale
                * Tensor::randn(&[2, num_pos_feats], (tch::Kind::Float, tch::Device::Cpu)),
        }
    }
    ///Positionally encode points that are normalized to [0,1].
    fn _pe_encoding(&self, coords: &Tensor) -> Tensor {
        let mut coords: Tensor = 2.0 * coords - 1.0;
        coords = coords.matmul(&self.positional_encoding_gaussian_matrix);
        coords = 2.0 * std::f32::consts::PI * coords;
        Tensor::cat(&[&coords.sin(), &coords.cos()], -1)
    }

    /// Generate positional encoding for a grid of the specified size.
    pub fn forward(&self, size: Size) -> Tensor {
        let Size(h, w) = size;
        let device = self.positional_encoding_gaussian_matrix.device();
        let grid = Tensor::ones(&[h, w], (tch::Kind::Float, device));
        let mut y_embed = grid.cumsum(0, tch::Kind::Float) - 0.5;
        let mut x_embed = grid.cumsum(1, tch::Kind::Float) - 0.5;
        y_embed = y_embed / h as f64;
        x_embed = x_embed / w as f64;
        let pe = self._pe_encoding(&Tensor::stack(&[x_embed, y_embed], -1));
        pe.permute(&[2, 0, 1])
    }

    /// Positionally encode points that are not normalized to [0,1].
    pub fn forward_with_coords(&self, coords_input: Tensor, image_size: Size) -> Tensor {
        let coords = coords_input.copy();
        coords
            .narrow(2, 0, 1)
            .copy_(&coords.narrow(2, 0, 1).div(image_size.1 as f64));
        coords
            .narrow(2, 1, 1)
            .copy_(&coords.narrow(2, 1, 1).div(image_size.0 as f64));

        self._pe_encoding(&coords.to_kind(tch::Kind::Float))
    }
}
