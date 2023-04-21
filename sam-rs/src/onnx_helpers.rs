use ndarray::ArrayD;
use onnxruntime::{
    environment::Environment, session::Session, tensor::OrtOwnedTensor, GraphOptimizationLevel,
};
use opencv::{
    imgcodecs, imgproc,
    prelude::{Mat, MatTraitConstManual},
};
use tch::Tensor;

use crate::sam_predictor::Size;

pub fn load_image(image_path: &str) -> (Tensor, Size) {
    let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR).unwrap();
    let mut rgb_image = Mat::default();
    imgproc::cvt_color(&image, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0).unwrap();
    let arr: ndarray::ArrayView3<u8> = rgb_image.try_as_array();
    let image = Tensor::try_from(arr).unwrap();
    let size = Size(image.size()[0], image.size()[1]);
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
pub struct OnnxInput {
    image_embedding: Tensor,
    coord: Tensor,
    label: Tensor,
    mask: Tensor,
    has_mask: Tensor,
    img_size: Tensor,
}
impl OnnxInput {
    pub fn new(
        image_embedding: Tensor,
        coord: Tensor,
        label: Tensor,
        mask: Tensor,
        has_mask: Tensor,
        img_size: Tensor,
    ) -> OnnxInput {
        Self {
            image_embedding,
            coord,
            label,
            mask,
            has_mask,
            img_size,
        }
    }
}
impl From<OnnxInput> for Vec<ArrayD<f32>> {
    fn from(input: OnnxInput) -> Self {
        let res: Vec<ArrayD<f32>> = vec![
            (&input.image_embedding).try_into().unwrap(),
            (&input.coord).try_into().unwrap(),
            (&input.label).try_into().unwrap(),
            (&input.mask).try_into().unwrap(),
            (&input.has_mask).try_into().unwrap(),
            (&input.img_size).try_into().unwrap(),
        ];
        res
    }
}
pub trait Inference {
    fn inference(&mut self, input: OnnxInput) -> (Tensor, Tensor, Tensor);
}
impl Inference for Session<'_> {
    fn inference(&mut self, input: OnnxInput) -> (Tensor, Tensor, Tensor) {
        let input: Vec<ArrayD<f32>> = input.into();
        let res: Vec<OrtOwnedTensor<f32, _>> = self.run(input).unwrap();
        (
            Tensor::try_from(res[0].clone()).unwrap(),
            Tensor::try_from(res[1].clone()).unwrap(),
            Tensor::try_from(res[2].clone()).unwrap(),
        )
    }
}

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
