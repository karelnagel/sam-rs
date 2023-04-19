extern crate ndarray;
use crate::ndarray as nd;

extern crate opencv;
use opencv::{core, imgcodecs, imgproc};

fn main() {
    // let sam = sam_rs::build_sam::build_sam_vit_h(None);
    // let mut predictor = sam_rs::sam_predictor::SamPredictor::new(sam);
    // // Image
    // let image = imgcodecs::imread("images/truck.jpg", imgcodecs::IMREAD_COLOR).unwrap();
    // let mut rgb_image = core::Mat::default();
    // imgproc::cvt_color(&image, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0).unwrap();

    // let arr: nd::ArrayView3<u8> = rgb_image.try_as_array();

    // predictor.set_image(arr.to_owned(), &sam_rs::sam_predictor::ImageFormat::RGB);

    // //Should be right
    // let point_coords = stack![Axis(0), input_point.mapv(|x| x as f32), array![[0.0, 0.0]]]
    //     .into_shape((1, 2, 2))
    //     .unwrap(); // Reshape to add an extra dimension at the beginning
    // let point_labels = stack![Axis(0), input_label.view(), array![-1.0].view()]
    //     .t()
    //     .to_owned();
    // let mask_input: Array<f32, Ix4> = Array::zeros((1, 1, 256, 256));
    // let has_mask_input: Array<f32, Ix1> = Array::zeros((1,));
    // let orig_im_size: Array<f32, Ix1> =
    //     Array::from_shape_vec((2,), vec![image.rows() as f32, image.cols() as f32]).unwrap();

    // let input_arrays: Vec<Array<f32, _>> = vec![
    //     image_embeddings.into_dyn(),
    //     point_coords.into_dyn(),
    //     point_labels.into_dyn(),
    //     mask_input.into_dyn(),
    //     has_mask_input.into_dyn(),
    //     orig_im_size.into_dyn(),
    // ];

    // let environment = Environment::builder().build().unwrap();

    // let mut session: onnxruntime::session::Session = environment
    //     .new_session_builder()
    //     .unwrap()
    //     .with_optimization_level(GraphOptimizationLevel::Basic)
    //     .unwrap()
    //     .with_number_threads(1)
    //     .unwrap()
    //     .with_model_from_file("sam.onnx")
    //     .unwrap();
    // let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_arrays).unwrap();
}
use opencv as cv2;
use opencv::prelude::*;

trait AsArray {
    fn try_as_array(&self) -> nd::ArrayView3<u8>;
}
impl AsArray for cv2::core::Mat {
    fn try_as_array(&self) -> nd::ArrayView3<u8> {
        let bytes = self.data_bytes().unwrap();
        let size = self.size().unwrap();
        let a = nd::ArrayView3::from_shape((size.height as usize, size.width as usize, 3), bytes)
            .unwrap();
        a
    }
}

pub fn ss() {
    let path = "./lena.png".to_string();
    let img = cv2::imgcodecs::imread(&path, cv2::imgcodecs::IMREAD_COLOR).unwrap();
    // error
    // let arr: nd::Array3<f32> = img.into_cv();
    let arr: nd::ArrayView3<u8> = img.try_as_array();
    println!("arr shape {:?}", arr.shape());
    // error
    // let img_rec: cv2::core::Mat = arr.try_into_cv().unwrap();
    cv2::highgui::imshow("img", &img).unwrap();
    // cv2::highgui::imshow("reconstruct", &img_rec);
    cv2::highgui::wait_key(0).unwrap();
}
