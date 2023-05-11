use std::{env, time::Instant};

use burn::{
    module::Module,
    record::{BinGzFileRecorder, DoublePrecisionSettings, Recorder},
};
use pyo3::prelude::*;
use sam_rs::{
    build_sam::SamVersion,
    python::module_to_file::module_to_file,
    tests::helpers::{get_python_sam, load_module, TestBackend},
};

fn python(variant: SamVersion, file: &str) -> PyResult<()> {
    Python::with_gil(|py| {
        let sam = get_python_sam(&py, variant, file)?;
        module_to_file(file, py, sam)?;
        Ok(())
    })
}
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        panic!("Usage: sam-convert <type> <file>");
    }
    let version = SamVersion::from_str(args[1].as_str());
    let file = args[2].as_str();
    let skip_python = args.len() == 4;

    let start = Instant::now();
    match skip_python {
        true => println!("Skipping python..."),
        false => {
            python(version, file).unwrap();
            println!("Python time: {:?}", start.elapsed());
        }
    }
    convert_sam(version, file);
    println!("Rust time: {:?}", start.elapsed());
}

fn convert_sam(sam: SamVersion, file: &str) {
    let mut sam = sam.build::<TestBackend>(None);
    println!("Loading module in rust...");
    sam = load_module(file, sam);
    println!("Saving module in rust...");

    let recorder = BinGzFileRecorder::<DoublePrecisionSettings>::default();
    recorder.record(sam.into_record(), file.into()).unwrap();
    println!("Done!")
}
