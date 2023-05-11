use burn::{
    module::Module,
    record::{BinGzFileRecorder, DoublePrecisionSettings, Recorder},
};
use sam_rs::{
    build_sam::SamVersion, python::recorder::load_module_from_python, tests::helpers::TestBackend,
};
use std::{env, time::Instant};

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
    sam = load_module_from_python(sam, version, file).unwrap();

    println!("Saving module in rust...");
    let recorder = BinGzFileRecorder::<DoublePrecisionSettings>::default();
    recorder.record(sam.into_record(), file.into()).unwrap();

    println!("Rust time: {:?}", start.elapsed());
}
