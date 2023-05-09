use std::{env, time::Instant};

use burn::{
    module::Module,
    record::{BinGzFileRecorder, DoublePrecisionSettings, Recorder},
};
use pyo3::prelude::*;
use sam_rs::{
    build_sam::BuildSam,
    python::module_to_file::module_to_file,
    tests::helpers::{load_module, TestBackend},
};

fn python(variant: &str, file: &str) -> PyResult<()> {
    Python::with_gil(|py| {
        let sam_model_registry = py
            .import("segment_anything.build_sam")?
            .getattr("sam_model_registry")?;
        let sam = sam_model_registry
            .get_item(variant)?
            .call1((file.to_string() + ".pth",))?;
        module_to_file(file, py, sam)?;
        Ok(())
    })
}
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        panic!("Usage: sam-convert <type> <file>");
    }
    let variant = args[1].as_str();
    let file = args[2].as_str();
    let skip_python = args.len() == 4;

    let sam = match variant {
        "test" => BuildSam::SamTest,
        "vit_h" => BuildSam::SamVitH,
        "vit_b" => BuildSam::SamVitB,
        "vit_l" => BuildSam::SamVitL,
        _ => panic!("Unknown variant: {}", variant),
    };
    let start = Instant::now();
    match skip_python {
        true => println!("Skipping python..."),
        false => {
            python(variant, file).unwrap();
            println!("Python time: {:?}", start.elapsed());
        }
    }
    convert_sam(sam, file);
    println!("Rust time: {:?}", start.elapsed());
}

fn convert_sam(sam: BuildSam, file: &str) {
    let mut sam = sam.build::<TestBackend>(None);
    println!("Loading module in rust...");
    sam = load_module(file, sam);
    println!("Saving module in rust...");

    let recorder = BinGzFileRecorder::<DoublePrecisionSettings>::default();
    recorder.record(sam.into_record(), file.into()).unwrap();
    println!("Done!")
}
