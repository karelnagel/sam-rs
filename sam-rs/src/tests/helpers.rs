use std::{collections::HashMap, fmt::Debug};

use burn::tensor::{backend::Backend, Tensor};
use burn_ndarray::NdArrayBackend;
use serde::{Deserialize, Serialize};

use crate::burn_helpers::TensorSlice;

use super::test_value::TestValue;

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct TestFile {
    pub values: HashMap<String, TestValue>,
}

pub type TestBackend = NdArrayBackend<f32>;
pub struct Test {
    file: TestFile,
}
impl Test {
    pub fn open(name: &str) -> Self {
        let path = format!("./test-files/{}.json", name);
        let file = std::fs::File::open(path).expect(format!("file {} not found", name).as_str());
        let reader = std::io::BufReader::new(file);
        let file = serde_json::from_reader(reader).unwrap();
        Self { file }
    }
    pub fn compare<T: Into<TestValue>>(&self, key: &str, value: T) {
        let file_value = self.file.values.get(key).unwrap().clone();
        assert_eq!(file_value, &value.into(), "key: '{}'", key);
        println!("{}: OK", key);
    }
}
fn random_slice(shape: &[usize], seed: usize) -> Vec<f32> {
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
