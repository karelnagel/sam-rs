use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::{
    burn_helpers::TensorSlice,
    modeling::common::activation::Activation,
    sam_predictor::{ImageFormat, Size},
};
#[derive(Deserialize, Serialize, PartialEq)]
pub struct TestTensor<T: PartialEq + Difference + std::fmt::Debug> {
    size: Vec<usize>,
    values: Vec<T>,
}
impl<T: PartialEq + Difference + std::fmt::Debug> TestTensor<T> {
    fn almost_equal(&self, other: &Self, threshold: f32) -> bool {
        if self.size != other.size {
            println!("TestTensor sizes don't match");
            return false;
        }
        let mut exact = 0;
        let mut almost = 0;
        let mut failed = 0;
        let mut max_diff: f32 = 0.0;
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            if a == b {
                exact += 1;
                continue;
            }
            let diff = a.difference(b);
            if diff <= threshold {
                almost += 1;
                continue;
            };
            max_diff = max_diff.max(diff);
            failed += 1;
        }
        let total = self.values.len();

        match failed {
            0 => true,
            _ => {
                println!(
                    "TestTensor::eq: exact: {}, almost: {}, failed: {}, total: {}! Max threshold: {}, current: {}",
                    exact, almost, failed, total,max_diff, threshold
                );
                false
            }
        }
    }
}

pub trait Difference {
    fn difference(&self, other: &Self) -> f32;
}
impl Difference for f32 {
    fn difference(&self, other: &Self) -> f32 {
        (self - other).abs() / self.abs().max(other.abs())
    }
}

impl Difference for i32 {
    fn difference(&self, other: &Self) -> f32 {
        (self - other).abs() as f32 / self.abs().max(other.abs()) as f32
    }
}
impl Difference for bool {
    fn difference(&self, other: &Self) -> f32 {
        match self == other {
            true => 0.0,
            false => 1.0,
        }
    }
}

impl<T: std::fmt::Debug + Clone + PartialEq + Difference> std::fmt::Debug for TestTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = &self.values.len();
        f.debug_struct("TestTensor")
            .field("size", &self.size)
            .field("start", &self.values[0..5].to_vec())
            .field("end", &self.values[len - 6..len - 1].to_vec())
            .finish()
    }
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub enum TestValue {
    TensorFloat(TestTensor<f32>),
    TensorBool(TestTensor<bool>),
    TensorInt(TestTensor<i32>),
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    List(Vec<usize>),
    ActivationType(Activation),
    Size(Size),
}
impl TestValue {
    pub fn almost_equal(&self, other: &Self, threshold: f32) -> bool {
        match (&self, &other) {
            (TestValue::TensorFloat(a), TestValue::TensorFloat(b)) => a.almost_equal(b, threshold),
            (TestValue::TensorBool(a), TestValue::TensorBool(b)) => a.almost_equal(b, threshold),
            (TestValue::TensorInt(a), TestValue::TensorInt(b)) => a.almost_equal(b, threshold),
            _ => self == other,
        }
    }
}
impl<B: Backend, const D: usize> From<Tensor<B, D>> for TestValue {
    fn from(tensor: Tensor<B, D>) -> Self {
        let (values, shape) = tensor.to_slice();
        TestValue::TensorFloat(TestTensor {
            size: shape.to_vec(),
            values,
        })
    }
}
impl<B: Backend, const D: usize> From<Tensor<B, D, Bool>> for TestValue {
    fn from(tensor: Tensor<B, D, Bool>) -> Self {
        let (values, shape) = tensor.to_slice();
        TestValue::TensorBool(TestTensor {
            size: shape.to_vec(),
            values,
        })
    }
}
impl<B: Backend, const D: usize> From<Tensor<B, D, Int>> for TestValue {
    fn from(tensor: Tensor<B, D, Int>) -> Self {
        let (values, shape) = tensor.to_slice();
        TestValue::TensorInt(TestTensor {
            size: shape.to_vec(),
            values,
        })
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
impl From<Activation> for TestValue {
    fn from(item: Activation) -> Self {
        TestValue::ActivationType(item)
    }
}
impl From<Vec<usize>> for TestValue {
    fn from(item: Vec<usize>) -> Self {
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
