use burn::{
    module::Module,
    record::{DoublePrecisionSettings, PrettyJsonFileRecorderSIMD, Recorder},
    tensor::backend::Backend,
};
use burn_tch::TchBackend;
use pyo3::{PyAny, PyResult, Python};

use crate::{build_sam::SamVersion, sam::Sam};

pub const TEST_ALMOST_THRESHOLD: f32 = 0.01;
pub type TestBackend = TchBackend<f64>;
pub const TEST_CHECKPOINT: &str = "../sam-convert/sam_test";
pub const TEST_SAM: SamVersion = SamVersion::Test;

pub fn get_sam<'a, T: Into<Option<SamVersion>>, C: Into<Option<&'a str>>>(
    version: T,
    checkpoint: C,
) -> Sam<TestBackend> {
    let version = version.into().unwrap_or(TEST_SAM);
    let checkpoint = checkpoint.into().unwrap_or(TEST_CHECKPOINT);
    let sam = version.build::<TestBackend>(Some(checkpoint));
    sam
}

pub fn get_python_sam<'a>(
    py: &'a Python,
    version: SamVersion,
    checkpoint: Option<&str>,
) -> PyResult<&'a PyAny> {
    let mut module = py
        .import("segment_anything.build_sam")?
        .getattr("sam_model_registry")?
        .get_item(version.to_str())?;

    match checkpoint {
        Some(checkpoint) => {
            let name = format!("{}.pth", checkpoint);
            module = module.call1((name,))?;
        }
        None => module = module.call0()?,
    }

    Ok(module)
}
pub fn get_python_test_sam<'a>(py: &'a Python) -> PyResult<&'a PyAny> {
    get_python_sam(&py, TEST_SAM, Some(TEST_CHECKPOINT))
}

pub fn load_module<B: Backend, D: Module<B>>(name: &str, module: D) -> D {
    let home = home::home_dir().unwrap();
    let path = home.join(format!("Documents/sam-models/{}.json", name));
    dbg!(&path);
    let recorder = PrettyJsonFileRecorderSIMD::<DoublePrecisionSettings>::default();
    let record: <D as Module<B>>::Record = recorder.load(path).unwrap();
    module.load_record(record)
}

pub fn save_module<B: Backend, M: Module<B>>(module: &M) {
    let recorder = PrettyJsonFileRecorderSIMD::<DoublePrecisionSettings>::default();
    recorder
        .record(module.clone().into_record(), "test.json".into())
        .unwrap();
}
