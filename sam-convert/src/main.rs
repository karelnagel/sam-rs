use std::{
    borrow::{Borrow, BorrowMut},
    env,
    time::Instant,
};
mod recorder;
mod update_tensor;
use burn::{
    module::Module,
    record::{BinGzFileRecorder, DoublePrecisionSettings, Recorder},
};
use pyo3::prelude::*;
use recorder::get_python_map;
use sam_rs::{
    build_sam::SamVersion,
    tests::helpers::{get_python_sam, TestBackend},
};

use crate::recorder::load_sam;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        panic!("Usage: sam-convert <type> <file>");
    }
    let version = args[1].as_str();
    let file = args[2].as_str();

    let version = SamVersion::from_str(version);

    let start = Instant::now();

    let mut sam = version.build::<TestBackend>(None);

    let res: PyResult<()> = Python::with_gil(|py| {
        // if test then won't load the weights.
        let python_sam = get_python_sam(
            &py,
            version,
            match version {
                SamVersion::Test => None,
                _ => Some(file),
            },
        )?;

        // Saves the module to pth, in test version.
        if version == SamVersion::Test {
            py.import("torch")?
                .call_method1("save", (python_sam, format!("{file}.pth")))?;
        }

        let map = get_python_map(python_sam).unwrap();
        println!("Python time: {:?}", start.elapsed());

        println!("Loading module in rust...");
        sam = load_sam(sam.clone(), map);

        Ok(())
    });
    println!("Saving module in rust...");
    let recorder = BinGzFileRecorder::<DoublePrecisionSettings>::default();
    recorder.record(sam.into_record(), file.into()).unwrap();

    println!("Rust time: {:?}", start.elapsed());

    res.unwrap();
}
