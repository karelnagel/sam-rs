// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use burn::tensor::Tensor;
use sam_rs::{
    build_sam::BuildSam,
    burn_helpers::{TensorHelpers, TensorSlice},
    sam_predictor::{ImageFormat, SamPredictor},
    tests::helpers::TestBackend,
};
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
// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn start_model(state: tauri::State<State>, window: Window, model: String, version: BuildSam) {
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
                        let (masks, _, _) = predictor.predict(
                            Some(input_point),
                            Some(input_label),
                            None,
                            None,
                            true,
                        );
                        let (slice, shape) = masks.to_slice();
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
fn is_model_active(state: tauri::State<State>) -> bool {
    state.0.lock().unwrap().sender.is_some()
}
#[tauri::command]
fn stop_model(state: tauri::State<State>) {
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
fn load_image(state: tauri::State<State>, path: String) {
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
fn predict_point(state: tauri::State<State>, coords: Vec<Vec<i32>>, labels: Vec<i32>) {
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

fn main() {
    tauri::Builder::default()
        .manage(State(Mutex::default()))
        .invoke_handler(tauri::generate_handler![
            start_model,
            stop_model,
            load_image,
            predict_point,
            is_model_active
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
