// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod sam;
use std::sync::Mutex;

use sam::State;

use crate::sam::{is_model_active, load_image, predict_point, start_model, stop_model};

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
