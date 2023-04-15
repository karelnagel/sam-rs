use std::{collections::HashMap, fmt::Debug};

use serde::{Deserialize, Serialize};
use tch::Tensor;

use crate::modeling::common::ActivationType;

#[derive(Debug, Deserialize, Serialize)]
pub struct TestFile {
    pub values: HashMap<String, TestValue>,
}
impl TestFile {
    pub fn open(name: &str) -> Self {
        let path = format!("./test-files/{}.json", name);
        let file = std::fs::File::open(path).expect(format!("file {} not found", name).as_str());
        let reader = std::io::BufReader::new(file);
        let test_file: Self = serde_json::from_reader(reader).unwrap();
        test_file
    }
    pub fn compare(&self, key: &str, value: &TestValue) {
        let file_value = self
            .values
            .get(key)
            .expect(format!("key {} not found", key).as_str());
        let res = match (file_value, value) {
            (TestValue::TensorFloat(val1), TestValue::TensorFloat(val2)) => compare(val1, val2),
            (TestValue::TensorInt(val1), TestValue::TensorInt(val2)) => compare(val1, val2),
            (TestValue::TensorBool(val1), TestValue::TensorBool(val2)) => compare(val1, val2),
            _ => {
                if file_value != value {
                    let error = format!(
                        "Key '{:?}' value is different: {:?} and {:?}",
                        key, file_value, value
                    );
                    Err(error)
                } else {
                    Ok(())
                }
            }
        };
        if let Err(error) = res {
            panic!("Key '{}' wasnt the same: {}", key, error);
        }
        println!("Key '{:?}' is correct", key);
    }
}
fn compare<T: std::cmp::PartialEq + MyTrait + Debug + Serialize>(
    val1: &TestTensor<T>,
    val2: &TestTensor<T>,
) -> Result<(), String> {
    if val1.size != val2.size {
        panic!(
            "Tensor size is different: {:?} and {:?}",
            val1.size, val2.size
        );
    }
    let res = val1.values.mean_error(&val2.values);
    if let Err(error) = res {
        let file = std::fs::File::create("error.json").unwrap();
        serde_json::to_writer_pretty(file, &(&val1.values, &val2.values)).unwrap();
        return Err(error);
    }
    Ok(())
}
#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub enum TestValue {
    TensorFloat(TestTensor<Vec<f64>>),
    TensorInt(TestTensor<Vec<i64>>),
    TensorBool(TestTensor<Vec<bool>>),
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    List(Vec<TestValue>),
    ActivationType(ActivationType),
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct TestTensor<T: MyTrait> {
    pub values: T,
    pub size: Vec<i64>,
}

pub trait ToTest {
    fn to_test(&self) -> TestValue;
}

impl ToTest for Tensor {
    fn to_test(&self) -> TestValue {
        let kind = self.kind();
        let size = self.size().to_vec();
        match kind {
            tch::Kind::Float => TestValue::TensorFloat(TestTensor {
                values: tensor_to_vec(self),
                size,
            }),
            tch::Kind::Int => TestValue::TensorInt(TestTensor {
                values: tensor_to_vec(self),
                size,
            }),
            tch::Kind::Bool => TestValue::TensorBool(TestTensor {
                values: tensor_to_vec(self),
                size,
            }),
            _ => panic!("Unsupported tensor kind: {:?}", kind),
        }
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
impl ToTest for ActivationType {
    fn to_test(&self) -> TestValue {
        TestValue::ActivationType(*self)
    }
}

pub fn tensor_to_vec<T>(tensor: &Tensor) -> Vec<T>
where
    T: tch::kind::Element,
    T: std::fmt::Debug,
{
    let flattened = tensor.flatten(0, -1);
    let flattened: Vec<T> = flattened.into();
    flattened
}

pub fn random_tensor(shape: &[i64]) -> Tensor {
    let mut value: f64 = 1.0;
    let mut values: Vec<f64> = vec![];
    for _ in 0..(shape.iter().product::<i64>()) {
        value += 1.0;
        values.push(value.round());
    }
    Tensor::of_slice(&values)
        .view(shape)
        .to_kind(tch::Kind::Float)
}

pub trait MyTrait {
    fn mean_error(&self, other: &Self) -> Result<(), String>;
}
const EPSILON: f64 = 0.001;

impl MyTrait for Vec<i64> {
    fn mean_error(&self, other: &Self) -> Result<(), String> {
        if self.len() != other.len() {
            return Err("Vectors must have the same length".to_string());
        }

        for (x, y) in self.iter().zip(other.iter()) {
            let diff = (x - y).abs() as f64;
            let avg = (x.abs() + y.abs()) as f64 / 2.0;
            let relative_diff = if avg == 0.0 { diff } else { diff / avg };
            if relative_diff > EPSILON {
                return Err(format!("Vectors are not equal: {:?} and {:?}", x, y));
            }
        }
        Ok(())
    }
}
impl MyTrait for Vec<f64> {
    fn mean_error(&self, other: &Self) -> Result<(), String> {
        if self.len() != other.len() {
            return Err("Vectors must have the same length".to_string());
        }
        for (x, y) in self.iter().zip(other.iter()) {
            let diff = (x - y).abs();
            let avg = (x.abs() + y.abs()) / 2.0;
            let relative_diff = if avg == 0.0 { diff } else { diff / avg };
            if relative_diff > EPSILON {
                return Err(format!("Vectors are not equal: {:?} and {:?}", x, y));
            }
        }
        Ok(())
    }
}
impl MyTrait for Vec<bool> {
    fn mean_error(&self, other: &Self) -> Result<(), String> {
        if self.len() != other.len() {
            return Err("Vectors must have the same length".to_string());
        }
        for (x, y) in self.iter().zip(other.iter()) {
            if x != y {
                return Err(format!("Vectors are not equal: {:?} and {:?}", x, y));
            }
        }
        Ok(())
    }
}
