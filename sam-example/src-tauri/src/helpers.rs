use std::sync::Arc;

use ndarray::{array, concatenate, s, Array, Array1, Array3, Axis};
use ort::{
    download::language::machine_comprehension::GPT2,
    tensor::{ort_owned_tensor::ViewHolder, DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, Session, SessionBuilder,
};
use sam_rs::{helpers::load_image_slice, utils::transforms::ResizeLongestSide};

pub fn start(encoder_path: &str, decoder_path: &str) -> OrtResult<(Session, Session)> {
    let environment = Arc::new(Environment::builder().build()?);

    let encoder_session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file(encoder_path)?;
    let decoder_session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file(decoder_path)?;

    Ok((encoder_session, decoder_session))
}
pub fn load_image(
    path: &str,
    encoder_session: &Session,
) -> OrtResult<ndarray::ArrayBase<ndarray::CowRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>> {
    let image_size = 1024;
    let transform = ResizeLongestSide::new(image_size);
    let (slice, shape) = load_image_slice(path);
    let image = Array1::from(slice).to_shape((shape.0, shape.1, 3)).unwrap();
    let input_image = image;
    let input_image: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>> =
        input_image
            .permuted_axes([2, 0, 1])
            .mapv_into_any(|x| x as f32);
    let pixel_mean = Array::from(vec![123.675, 116.28, 103.53])
        .to_shape((3, 1, 1))
        .unwrap();
    let pixel_std = Array::from(vec![58.395, 57.12, 57.375])
        .to_shape((3, 1, 1))
        .unwrap();
    let x = (input_image - pixel_mean) / pixel_std;
    let h = x.shape()[1];
    let w = x.shape()[2];
    let padH = image_size - h;
    let padW = image_size - w;
    // let x = x.pad((0,padw,0,padH));
    let outputs = encoder_session.run([InputTensor::from_array(x.into_dyn())])?;
    let array: ViewHolder<f32, ndarray::Dim<ndarray::IxDynImpl>> = outputs[0].try_extract()?.view();
    let slice: Vec<f32> = array.iter().map(|x| *x).collect();
    let shape = array.shape();
    let array = Array::from(slice).to_shape(shape).unwrap();
    Ok(array)
}
