use onnxruntime::{
    environment::Environment,
    ndarray::{array, stack, Array, Axis, Ix1, Ix2, Ix3, Ix4},
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel,
};

extern crate opencv;

use opencv::{core, imgcodecs, imgproc, prelude::MatTraitConst};
fn main() {
    // Image
    let image = imgcodecs::imread("images/truck.jpg", imgcodecs::IMREAD_COLOR).unwrap();
    let mut rgb_image = core::Mat::default();
    imgproc::cvt_color(&image, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0).unwrap();

    let input_point = array![[500, 875]];
    let input_label = array![1.0]; // Ensure element type is f32

    // Wrong
    let image_embeddings: Array<f32, Ix4> = Array::zeros((1, 256, 64, 64));
    let point_coords: Array<f32, Ix3> = Array::zeros((1, 2, 2));

    //Should be right
    let point_labels = stack![Axis(0), input_label.view(), array![-1.0].view()]
        .t()
        .to_owned();
    let mask_input: Array<f32, Ix4> = Array::zeros((1, 1, 256, 256));
    let has_mask_input: Array<f32, Ix1> = Array::zeros((1,));
    let orig_im_size: Array<f32, Ix1> =
        Array::from_shape_vec((2,), vec![image.rows() as f32, image.cols() as f32]).unwrap();

    let input_arrays: Vec<Array<f32, _>> = vec![
        image_embeddings.into_dyn(),
        point_coords.into_dyn(),
        point_labels.into_dyn(),
        mask_input.into_dyn(),
        has_mask_input.into_dyn(),
        orig_im_size.into_dyn(),
    ];

    let environment = Environment::builder().build().unwrap();

    let mut session: onnxruntime::session::Session = environment
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_number_threads(1)
        .unwrap()
        .with_model_from_file("sam.onnx")
        .unwrap();
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_arrays).unwrap();
}
