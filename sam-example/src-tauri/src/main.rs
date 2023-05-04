// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use burn::tensor::Tensor;
use sam_rs::{
    build_sam::build_sam_vit_h,
    burn_helpers::{TensorHelpers, TensorSlice},
    sam_predictor::SamPredictor,
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
fn start_model(state: tauri::State<State>, window: Window) {
    let rx = {
        let mut app_state = state.0.lock().unwrap();
        if app_state.sender.is_some() {
            print!("Model already running!");
            return;
        }
        let (tx, rx) = flume::unbounded::<Props>();
        app_state.sender = Some(tx);
        rx
    };

    std::thread::spawn(move || {
        let checkpoint = Some("sam_vit_h");
        let sam = build_sam_vit_h::<TestBackend>(checkpoint);
        let mut predictor = SamPredictor::new(sam);

        loop {
            let props = rx.recv().unwrap();
            match props {
                Props::Stop => break,
                Props::LoadImage(path) => {
                    let (image, _) = sam_rs::helpers::load_image(&path);
                    predictor.set_image(image, sam_rs::sam_predictor::ImageFormat::RGB);
                }
                Props::PredictPoint(coords, labels) => {
                    assert_eq!(coords.len(), labels.len() * 2);
                    let input_point = Tensor::of_slice(coords.clone(), [coords.len()])
                        .reshape_max([usize::MAX, 2]);
                    let input_label = Tensor::of_slice(labels.clone(), [labels.len()]);
                    let (masks, _, _) =
                        predictor.predict(Some(input_point), Some(input_label), None, None, true);
                    //Todo send result
                    print!("masks: {:?}", masks.dims());
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    });

    //Todo save predicotr
}

#[tauri::command]
fn is_model_running(state: tauri::State<State>) -> bool {
    state.0.lock().unwrap().sender.is_some()
}
#[tauri::command]
fn stop_model(state: tauri::State<State>) {
    let mut app_state = state.0.lock().unwrap();
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
        app_state.sender = None;
    }
}

#[tauri::command]
fn predict_point(state: tauri::State<State>, coords: Vec<i32>, labels: Vec<i32>) {
    assert_eq!(coords.len(), labels.len() * 2);
    let mut app_state = state.0.lock().unwrap();
    if app_state.sender.is_some() {
        app_state
            .sender
            .as_mut()
            .unwrap()
            .send(Props::PredictPoint(coords, labels))
            .unwrap();
        app_state.sender = None;
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
            is_model_running
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
