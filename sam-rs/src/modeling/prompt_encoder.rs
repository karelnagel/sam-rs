use std::ops::Div;

use tch::{nn, Device, Tensor};

use crate::sam_predictor::Size;

use super::mask_decoder::Activation;

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
        embed_dim: i64,
        image_embedding_size: Size,
        input_image_size: Size,
        mask_in_chans: i64,
        activation: Activation,
    ) -> Self {
        let pe_layer = PositionEmbeddingRandom::new(Some(embed_dim / 2), None);
        let num_point_embeddings: i64 = 4; // pos/neg point + 2 box corners
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let point_embeddings = (0..num_point_embeddings)
            .map(|_| nn::embedding(&vs.root(), 1, embed_dim, Default::default()))
            .collect::<Vec<_>>();
        let not_a_point_embed = nn::embedding(&vs.root(), 1, embed_dim, Default::default());
        let mask_input_size = Size(4 * image_embedding_size.0, 4 * image_embedding_size.1);
        let mask_downscaling = nn::seq()
            .add(nn::conv2d(
                &vs.root(),
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
                &vs.root(),
                vec![mask_in_chans / 4],
                Default::default(),
            ))
            // .add(activation.build()) //Todo
            .add(nn::conv2d(
                &vs.root(),
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
                &vs.root(),
                vec![mask_in_chans / 8],
                Default::default(),
            ));
        // .add(activation.build());
        let no_mask_embed = nn::embedding(&vs.root(), 1, embed_dim, Default::default());
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
        unimplemented!()
        // points = points + 0.5  # Shift to center of pixel
        // if pad:
        //     padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
        //     padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
        //     points = torch.cat([points, padding_point], dim=1)
        //     labels = torch.cat([labels, padding_label], dim=1)
        // point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        // point_embedding[labels == -1] = 0.0
        // point_embedding[labels == -1] += self.not_a_point_embed.weight
        // point_embedding[labels == 0] += self.point_embeddings[0].weight
        // point_embedding[labels == 1] += self.point_embeddings[1].weight
        // return point_embedding
    }

    ///Embeds box prompts.
    fn _embed_boxes(&self, boxes: Tensor) -> Tensor {
        unimplemented!()
        // boxes = boxes + 0.5  # Shift to center of pixel
        // coords = boxes.reshape(-1, 2, 2)
        // corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        // corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        // corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        // return corner_embedding
    }

    ///Embeds mask inputs.
    fn _embed_masks(&self, masks: Tensor) -> Tensor {
        unimplemented!()
        // mask_embedding = self.mask_downscaling(masks)
        // return mask_embedding
    }

    /// Gets the batch size of the output given the batch size of the input prompts.
    fn _get_batch_size(
        &self,
        points: Option<(Tensor, Tensor)>,
        boxes: Option<Tensor>,
        masks: Option<Tensor>,
    ) -> i64 {
        unimplemented!()
        // if points is not None:
        //     return points[0].shape[0]
        // elif boxes is not None:
        //     return boxes.shape[0]
        // elif masks is not None:
        //     return masks.shape[0]
        // else:
        //     return 1
    }

    fn _get_device(&self) -> Device {
        unimplemented!()

        //return self.point_embeddings[0].weight.device
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
        unimplemented!()
        // bs = self._get_batch_size(points, boxes, masks)
        // sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        // if points is not None:
        //     coords, labels = points
        //     point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
        //     sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        // if boxes is not None:
        //     box_embeddings = self._embed_boxes(boxes)
        //     sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        // if masks is not None:
        //     dense_embeddings = self._embed_masks(masks)
        // else:
        //     dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
        //         bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        //     )

        // return sparse_embeddings, dense_embeddings
    }
}

/// Positional encoding using random spatial frequencies.
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
        let grid = Tensor::ones(&[h as i64, w as i64], (tch::Kind::Float, device));
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
