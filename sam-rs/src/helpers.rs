use burn::tensor::{backend::Backend, Tensor};
use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel};
use opencv::{
    imgcodecs, imgproc,
    prelude::{Mat, MatTraitConstManual},
};

use crate::{burn_helpers::TensorSlice, sam_predictor::Size};

pub fn load_image<B: Backend>(image_path: &str) -> (Tensor<B, 3>, Size) {
    let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR).unwrap();
    let mut rgb_image = Mat::default();
    imgproc::cvt_color(&image, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0).unwrap();
    let arr: ndarray::ArrayView3<u8> = rgb_image.try_as_array();
    let slice: Vec<f32> = vec![];
    let shape = [arr.shape()[0], arr.shape()[1], arr.shape()[2]];
    let image = Tensor::of_slice(slice, shape);
    let size = Size(image.shape().dims[0], image.shape().dims[1]);
    (image, size)
}

pub fn get_ort_env() -> Environment {
    Environment::builder().build().unwrap()
}
pub fn get_ort_session<'a>(pth: &'a str, env: &'a Environment) -> Session<'a> {
    let session: Session = env
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_number_threads(1)
        .unwrap()
        .with_model_from_file(pth)
        .unwrap();
    session
}
// pub struct OnnxInput<B: Backend> {
//     image_embedding: Tensor<B, 4>,
//     coord: Tensor<B, 2>,
//     label: Tensor<B, 1>,
//     mask: Tensor<B, 2>,
//     has_mask: Tensor<B, 1>,
//     img_size: Tensor<B, 1>,
// }
// impl<B: Backend> OnnxInput<B> {
//     pub fn new(
//         image_embedding: Tensor<B, 4>,
//         coord: Tensor<B, 2>,
//         label: Tensor<B, 1>,
//         mask: Tensor<B, 4>,
//         has_mask: Tensor<B, 1>,
//         img_size: Tensor<B, 1>,
//     ) -> Self {
//         Self {
//             image_embedding,
//             coord,
//             label,
//             mask,
//             has_mask,
//             img_size,
//         }
//     }
// }
// impl<B: Backend> From<OnnxInput<B>> for Vec<ArrayD<f32>> {
//     fn from(input: OnnxInput<B>) -> Self {
//         let res: Vec<ArrayD<f32>> = vec![
//             (&input.image_embedding).try_into().unwrap(),
//             (&input.coord).unwrap(),
//             (&input.label).try_into().unwrap(),
//             (&input.mask).try_into().unwrap(),
//             (&input.has_mask).try_into().unwrap(),
//             (&input.img_size).try_into().unwrap(),
//         ];
//         res
//     }
// }
// pub trait Inference {
//     fn inference<B: Backend>(
//         &mut self,
//         input: OnnxInput<B>,
//     ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>);
// }
// impl Inference for Session<'_> {
//     fn inference<B: Backend>(
//         &mut self,
//         input: OnnxInput<B>,
//     ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
//         let input: Vec<ArrayD<f32>> = input.into();
//         let res: Vec<OrtOwnedTensor<f32, _>> = self.run(input).unwrap();
//         (
//             Tensor::try_from(res[0].clone()).unwrap(),
//             Tensor::try_from(res[1].clone()).unwrap(),
//             Tensor::try_from(res[2].clone()).unwrap(),
//         )
//     }
// }

pub trait AsArray {
    fn try_as_array(&self) -> ndarray::ArrayView3<u8>;
}
impl AsArray for Mat {
    fn try_as_array(&self) -> ndarray::ArrayView3<u8> {
        let bytes = self.data_bytes().unwrap();
        let size = self.size().unwrap();
        let a =
            ndarray::ArrayView3::from_shape((size.height as usize, size.width as usize, 3), bytes)
                .unwrap();
        a
    }
}