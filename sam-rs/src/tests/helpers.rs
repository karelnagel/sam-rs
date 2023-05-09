use std::{collections::HashMap, fmt::Debug};

use burn::{
    module::Module,
    record::{DoublePrecisionSettings, PrettyJsonFileRecorderSIMD, Recorder},
    tensor::{backend::Backend, Int, Tensor},
};
use burn_tch::TchBackend;
use serde::{Deserialize, Serialize};

use crate::{build_sam::BuildSam, burn_helpers::TensorHelpers, sam::Sam};

use super::test_value::TestValue;

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct TestFile {
    pub values: HashMap<String, TestValue>,
}
pub const TEST_CHECKPOINT: Option<&str> = Some("../sam-convert/sam_test");
pub const TEST_SAM: BuildSam = BuildSam::SamTest;
pub fn get_test_sam() -> Sam<TestBackend> {
    let sam = TEST_SAM.build::<TestBackend>(TEST_CHECKPOINT);
    sam
}
pub const TEST_ALMOST_THRESHOLD: f32 = 0.01;
pub type TestBackend = TchBackend<f64>;
pub struct Test {
    name: String,
    file: TestFile,
}
impl Test {
    pub fn open(name: &str) -> Self {
        let home = home::home_dir().unwrap();
        let path = home.join(format!("Documents/test-outputs/{}.json", name));
        dbg!(&path);
        let file = std::fs::File::open(&path).expect(format!("file {} not found", name).as_str());
        let reader = std::io::BufReader::new(file);
        let file =
            simd_json::from_reader(reader).expect(format!("file {:?} not valid", path).as_str());
        Self {
            file,
            name: name.into(),
        }
    }
    pub fn equal<T: Into<TestValue>>(&self, key: &str, value: T) {
        let file_value = self.file.values.get(key).unwrap();
        assert_eq!(
            file_value,
            &value.into(),
            "key '{}' failed in '{}' test",
            key,
            self.name
        );
        println!("{}: OK", key);
    }
    pub fn almost_equal<T: Into<TestValue>, K: Into<Option<f32>>>(
        &self,
        key: &str,
        value: T,
        threshold: K,
    ) {
        let threshold = threshold.into().unwrap_or(TEST_ALMOST_THRESHOLD);
        let file_value = self.file.values.get(key).unwrap();
        let value: TestValue = value.into();
        if !file_value.almost_equal(&value, threshold) {
            println!("left: {:?}", file_value);
            println!("right: {:?}", value);
            panic!("key '{}' failed in '{}' test", key, self.name);
        }

        println!("{}: OK", key);
    }
}

pub fn random_slice(shape: &[usize], seed: usize) -> Vec<f32> {
    let n = shape.iter().product::<usize>();
    let a = 3_usize;
    let c = 23_usize;
    let m = 2_usize.pow(4);

    let mut result = Vec::new();
    let mut x = seed;
    for _ in 0..n {
        x = (a.wrapping_mul(x).wrapping_add(c)) % m;
        result.push(x as f32 / m as f32); // Normalize the result to [0, 1]
    }
    result
}

pub fn random_tensor<B: Backend, const D: usize>(shape: [usize; D], seed: usize) -> Tensor<B, D> {
    let slice = random_slice(&shape, seed);
    Tensor::of_slice(slice, shape)
}
pub fn random_tensor_int<B: Backend, const D: usize>(
    shape: [usize; D],
    seed: usize,
    multiply: f32,
) -> Tensor<B, D, Int> {
    let slice = random_slice(&shape, seed);
    let slice = slice.iter().map(|x| (*x * multiply) as i32).collect();
    Tensor::of_slice(slice, shape)
}

pub fn load_module<B: Backend, D: Module<B>>(name: &str, module: D) -> D {
    let home = home::home_dir().unwrap();
    let path = home.join(format!("Documents/sam-models/{}.json", name));
    dbg!(&path);
    let recorder = PrettyJsonFileRecorderSIMD::<DoublePrecisionSettings>::default();
    let record = recorder.load(path).unwrap();
    module.load_record(record)
}

pub fn save_module<B: Backend, M: Module<B>>(module: &M) {
    let recorder = PrettyJsonFileRecorderSIMD::<DoublePrecisionSettings>::default();
    recorder
        .record(module.clone().into_record(), "test.json".into())
        .unwrap();
}
