use std::{env, process::Command, time::Instant};

use burn::{
    module::Module,
    record::{Record, SentitiveCompactRecordSettings},
};
use sam_rs::{
    build_sam::BuildSam,
    tests::helpers::{load_module, TestBackend},
};
fn python(variant: &str, file: &str) {
    let mut child = Command::new("python")
        .arg("../convert.py")
        .arg(variant)
        .arg(file)
        .spawn()
        .expect("python command failed to start");

    let status = child.wait().expect("Failed to wait for the python command");
    assert!(status.success());
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
            python(variant, file);
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
    sam.into_record()
        .record::<SentitiveCompactRecordSettings>(file.into())
        .unwrap();
    println!("Done!")
}
