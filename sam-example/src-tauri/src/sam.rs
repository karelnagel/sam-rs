use burn::tensor::Tensor;
use sam_rs::{
    build_sam::SamVersion,
    burn_helpers::TensorHelpers,
    sam_predictor::{ImageFormat, SamPredictor},
    tests::helpers::TestBackend,
};
use std::sync::Mutex;
use tauri::Window;
pub enum Props {
    Stop,
    LoadImage(String),
    PredictPoint(Vec<i32>, Vec<i32>),
}
pub struct AppState {
    pub sender: Option<flume::Sender<Props>>,
}
impl Default for AppState {
    fn default() -> Self {
        Self { sender: None }
    }
}

pub struct State(pub Mutex<AppState>);
#[tauri::command]
pub fn start_model(state: tauri::State<State>, window: Window, model: String, version: SamVersion) {
    let mut app_state = state.0.lock().unwrap();
    if app_state.sender.is_some() {
        print!("Model already running!");
        return;
    }
    let (tx, rx) = flume::unbounded::<Props>();
    app_state.sender = Some(tx);

    std::thread::spawn(move || {
        println!(
            "Starting model, with version: {:?} and path: {}",
            version, model
        );
        let checkpoint = Some(model.as_str());
        let sam = version.build::<TestBackend>(checkpoint);
        let mut predictor = SamPredictor::new(sam);
        println!("Model started!");
        loop {
            match rx.recv() {
                Ok(props) => match props {
                    Props::Stop => {
                        println!("Stopping model...");
                        break;
                    }
                    Props::LoadImage(path) => {
                        println!("Loading image: {}", path);
                        let (image, _) = sam_rs::helpers::load_image(&path);
                        predictor.set_image(image, ImageFormat::RGB);
                        println!("Image loaded!")
                    }
                    Props::PredictPoint(coords, labels) => {
                        println!("Predicting point...");
                        assert_eq!(coords.len(), labels.len() * 2);
                        let input_point = Tensor::of_slice(coords.clone(), [coords.len()])
                            .reshape_max([usize::MAX, 2]);
                        let input_label = Tensor::of_slice(labels.clone(), [labels.len()]);
                        let (masks, _, _, _) = predictor.predict(
                            Some(input_point),
                            Some(input_label),
                            None,
                            None,
                            true,
                        );
                        let shape = masks.dims();
                        let slice: Vec<bool> = masks.to_data().value.into_iter().collect();

                        window.emit("masks", (slice, shape)).unwrap();
                        println!("Point predicted!");
                    }
                },
                Err(e) => {
                    println!("Error: {}", e);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        println!("Model stopped!")
    });
}

#[tauri::command]
pub fn is_model_active(state: tauri::State<State>) -> bool {
    state.0.lock().unwrap().sender.is_some()
}
#[tauri::command]
pub fn stop_model(state: tauri::State<State>) {
    let mut app_state = state.0.lock().expect("already stopped");
    if app_state.sender.is_some() {
        app_state
            .sender
            .as_mut()
            .unwrap()
            .send(Props::Stop)
            .unwrap();
        app_state.sender = None;
    }
}

#[tauri::command]
pub fn load_image(state: tauri::State<State>, path: String) {
    let mut app_state = state.0.lock().unwrap();
    if app_state.sender.is_some() {
        app_state
            .sender
            .as_mut()
            .unwrap()
            .send(Props::LoadImage(path))
            .unwrap();
    }
}

#[tauri::command]
pub fn predict_point(state: tauri::State<State>, coords: Vec<Vec<i32>>, labels: Vec<i32>) {
    assert_eq!(coords.len(), labels.len());
    let mut app_state = state.0.lock().unwrap();
    if app_state.sender.is_some() {
        let coords = coords.iter().flatten().map(|x| *x).collect::<Vec<_>>();
        app_state
            .sender
            .as_mut()
            .unwrap()
            .send(Props::PredictPoint(coords, labels))
            .unwrap();
    }
}
