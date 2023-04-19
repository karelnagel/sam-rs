use std::{collections::HashMap, fmt::Debug};

use serde::{Deserialize, Serialize};
use tch::Tensor;

use crate::{
    modeling::common::activation::ActivationType,
    sam_predictor::{ImageFormat, Size},
};

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
    pub fn compare<T: Into<TestValue>>(&self, key: &str, value: T) {
        self.comp(key, &value.into(), false)
    }
    pub fn compare_only_size<T: Into<TestValue>>(&self, key: &str, value: T) {
        self.comp(key, &value.into(), true)
    }
    fn comp(&self, key: &str, value: &TestValue, only_size: bool) {
        let file_value = self
            .values
            .get(key)
            .expect(format!("key {} not found", key).as_str());
        let res = match (file_value, value) {
            (TestValue::TensorFloat(val1), TestValue::TensorFloat(val2)) => {
                compare(val1, val2, only_size)
            }
            (TestValue::TensorInt(val1), TestValue::TensorInt(val2)) => {
                compare(val1, val2, only_size)
            }
            (TestValue::TensorBool(val1), TestValue::TensorBool(val2)) => {
                compare(val1, val2, only_size)
            }
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
fn compare<T: std::cmp::PartialEq + IsSame + Debug + Serialize>(
    val1: &TestTensor<T>,
    val2: &TestTensor<T>,
    only_size: bool,
) -> Result<(), String> {
    if val1.size != val2.size {
        panic!(
            "Tensor size is different: {:?} and {:?}",
            val1.size, val2.size
        );
    }
    if !only_size {
        let res = val1.values.is_same(&val2.values);
        if let Err(error) = res {
            let file = std::fs::File::create("error.json").unwrap();
            serde_json::to_writer_pretty(file, &(&val1.values, &val2.values)).unwrap();
            return Err(error);
        }
    }
    Ok(())
}
#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub enum TestValue {
    TensorFloat(TestTensor<Vec<f64>>),
    TensorInt(TestTensor<Vec<i64>>),
    TensorBool(TestTensor<Vec<bool>>),
    TensorUint8(TestTensor<Vec<u8>>),
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    List(Vec<i64>),
    ActivationType(ActivationType),
    Size(Size),
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct TestTensor<T: IsSame> {
    pub values: T,
    pub size: Vec<i64>,
}

impl From<Tensor> for TestValue {
    fn from(tensor: Tensor) -> Self {
        let kind: tch::Kind = tensor.kind();
        let size = tensor.size().to_vec();
        match kind {
            tch::Kind::Float => TestValue::TensorFloat(TestTensor {
                values: tensor_to_vec(&tensor),
                size,
            }),
            tch::Kind::Int => TestValue::TensorInt(TestTensor {
                values: tensor_to_vec(&tensor),
                size,
            }),
            tch::Kind::Bool => TestValue::TensorBool(TestTensor {
                values: tensor_to_vec(&tensor),
                size,
            }),
            tch::Kind::Uint8 => TestValue::TensorUint8(TestTensor {
                values: tensor_to_vec(&tensor),
                size,
            }),
            _ => panic!("Unsupported tensor kind: {:?}", kind),
        }
    }
}
impl From<f64> for TestValue {
    fn from(item: f64) -> Self {
        TestValue::Float(item)
    }
}
impl From<Size> for TestValue {
    fn from(item: Size) -> Self {
        TestValue::Size(item)
    }
}
impl From<i64> for TestValue {
    fn from(item: i64) -> Self {
        TestValue::Int(item)
    }
}
impl From<usize> for TestValue {
    fn from(item: usize) -> Self {
        TestValue::Int(item as i64)
    }
}
impl From<String> for TestValue {
    fn from(item: String) -> Self {
        TestValue::String(item.to_string())
    }
}
impl From<bool> for TestValue {
    fn from(item: bool) -> Self {
        TestValue::Bool(item)
    }
}
impl From<ActivationType> for TestValue {
    fn from(item: ActivationType) -> Self {
        TestValue::ActivationType(item)
    }
}
impl From<Vec<i64>> for TestValue {
    fn from(item: Vec<i64>) -> Self {
        TestValue::List(item)
    }
}
impl From<ImageFormat> for TestValue {
    fn from(item: ImageFormat) -> Self {
        TestValue::String(
            match item {
                ImageFormat::BGR => "BGR",
                ImageFormat::RGB => "RGB",
            }
            .to_string(),
        )
    }
}

pub fn tensor_to_vec<T>(tensor: &Tensor) -> Vec<T>
where
    T: tch::kind::Element,
    T: std::fmt::Debug,
{
    let flattened = tensor.flatten(0, -1);
    flattened.into()
}
fn random_slice(shape: &[i64], seed: u64) -> Vec<f64> {
    let n = shape.iter().product::<i64>();
    let a: u64 = 3;
    let c: u64 = 23;
    let m: u64 = 2u64.pow(32);

    let mut result = Vec::new();
    let mut x = seed;
    for _ in 0..n {
        x = (a.wrapping_mul(x).wrapping_add(c)) % m;
        result.push(x as f64 / m as f64); // Normalize the result to [0, 1]
    }
    result
}

pub fn random_tensor(shape: &[i64], seed: u64) -> Tensor {
    let slice = random_slice(shape, seed);
    Tensor::of_slice(&slice)
        .view(shape)
        .to_kind(tch::Kind::Float)
}

pub trait IsSame {
    fn is_same(&self, other: &Self) -> Result<(), String>;
}
const RELATIVE_TOLERANCE: f64 = 0.004;

impl IsSame for Vec<i64> {
    fn is_same(&self, other: &Self) -> Result<(), String> {
        if self.len() != other.len() {
            return Err("Vectors must have the same length".to_string());
        }
        for (i, (a, b)) in self.iter().zip(other.iter()).enumerate() {
            // Calculate the absolute difference between the two elements.
            let diff = (*a as f64 - *b as f64).abs();
            // Calculate the maximum absolute value of the two elements.
            let max_abs_value = a.abs().max(b.abs()) as f64;
            // Calculate the relative tolerance based on the maximum absolute value.
            let tolerance = RELATIVE_TOLERANCE * max_abs_value;

            // Check if the difference exceeds the relative tolerance.
            if diff > tolerance {
                return Err(format!(
                    "Elements at index {} differ by more than the allowed tolerance",
                    i
                ));
            }
        }
        Ok(())
    }
}

impl IsSame for Vec<f64> {
    fn is_same(&self, other: &Self) -> Result<(), String> {
        if self.len() != other.len() {
            return Err("Vectors must have the same length".to_string());
        }
        for (i, (a, b)) in self.iter().zip(other.iter()).enumerate() {
            // Calculate the absolute difference between the two elements.
            let diff = (*a as f64 - *b as f64).abs();
            // Calculate the maximum absolute value of the two elements.
            let max_abs_value = a.abs().max(b.abs()) as f64;
            // Calculate the relative tolerance based on the maximum absolute value.
            let tolerance = RELATIVE_TOLERANCE * max_abs_value;

            // Check if the difference exceeds the relative tolerance.
            if diff > tolerance {
                return Err(format!(
                    "Elements at index {} differ by more than the allowed tolerance, a: {}, b: {}",
                    i, a, b
                ));
            }
        }
        Ok(())
    }
}
impl IsSame for Vec<bool> {
    fn is_same(&self, other: &Self) -> Result<(), String> {
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

impl IsSame for Vec<u8> {
    fn is_same(&self, other: &Self) -> Result<(), String> {
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
