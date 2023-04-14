use std::collections::HashMap;

use md5::{Digest, Md5};
use serde::{Deserialize, Serialize};
use tch::Tensor;

#[derive(Debug, Deserialize, Serialize)]
pub struct TestFile {
    pub name: String,
    pub values: HashMap<String, TestValue>,
}
impl TestFile {
    pub fn open(name: &str) -> Self {
        let path = format!("./test-files/{}.json", name);
        println!("path: {:?}", path);
        let file = std::fs::File::open(path).unwrap();
        let reader = std::io::BufReader::new(file);
        let test_file: Self = serde_json::from_reader(reader).unwrap();
        test_file
    }
    pub fn compare(&self, key: &str, value: &TestValue) {
        let file_value = self
            .values
            .get(key)
            .expect(format!("key {} not found", key).as_str());
        if file_value == value {
            println!("true");
        } else {
            let error = format!("file value: {:?} != value: {:?}", file_value, value);
            panic!("{}", error);
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub enum TestValue {
    Tensor(TestTensor),
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    List(Vec<TestValue>),
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct TestTensor {
    pub hash: String,
    pub size: Vec<i64>,
}

pub trait ToTest {
    fn to_test(&self) -> TestValue;
}

impl ToTest for Tensor {
    fn to_test(&self) -> TestValue {
        let hash = hash_tensor(self);
        let size = self.size();
        TestValue::Tensor(TestTensor { hash, size })
    }
}
impl ToTest for f64 {
    fn to_test(&self) -> TestValue {
        TestValue::Float(*self)
    }
}
impl ToTest for i64 {
    fn to_test(&self) -> TestValue {
        TestValue::Int(*self)
    }
}
impl ToTest for String {
    fn to_test(&self) -> TestValue {
        TestValue::String(self.to_string())
    }
}
impl ToTest for bool {
    fn to_test(&self) -> TestValue {
        TestValue::Bool(*self)
    }
}

pub fn hash(value: String) -> String {
    let mut hasher = Md5::new();
    hasher.update(value.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

pub fn hash_tensor(tensor: &Tensor) -> String {
    let flattened = tensor.flatten(0, -1);
    let value = format!("{:?}", flattened);
    let hash = hash(value);
    println!("hash: {:?}", hash);
    return hash;
}

pub fn random_tensor(size: &[i64]) -> Tensor {
    tch::manual_seed(4);
    Tensor::randn(size, (tch::Kind::Float, tch::Device::Cpu))
}
