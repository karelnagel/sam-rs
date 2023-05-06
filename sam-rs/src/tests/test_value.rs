use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::{
    burn_helpers::TensorSlice,
    modeling::common::activation::Activation,
    sam_predictor::{ImageFormat, Size},
};
#[derive(Deserialize, Serialize)]
pub struct TestTensor<T: PartialEq> {
    size: Vec<usize>,
    values: Vec<T>,
}
const EQUALITY_THRESHOLD: f32 = 0.1;
impl PartialEq for TestTensor<f32> {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        for (i, (a, b)) in self.values.iter().zip(other.values.iter()).enumerate() {
            let a_abs = a.abs();
            let b_abs = b.abs();
            let low = a_abs - (a_abs * EQUALITY_THRESHOLD);
            let high = a_abs + (a_abs * EQUALITY_THRESHOLD);
            if !(low <= b_abs && b_abs <= high) {
                let diff = (a - b).abs() / a_abs;
                println!(
                    "TestTensor::eq: {} != {} at index {}, current threshold {}, but needed {}",
                    a, b, i, EQUALITY_THRESHOLD, diff
                );
                return false;
            }
        }
        true
    }
}
impl PartialEq for TestTensor<i32> {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        for (i, (a, b)) in self.values.iter().zip(other.values.iter()).enumerate() {
            let a = *a as f32;
            let b = *b as f32;
            let a_abs = a.abs();
            let b_abs = b.abs();
            let low = a_abs - (a_abs * EQUALITY_THRESHOLD);
            let high = a_abs + (a_abs * EQUALITY_THRESHOLD);
            if !(low <= b_abs && b_abs <= high) {
                let diff = (a - b).abs() / a_abs;
                println!(
                    "TestTensor::eq: {} != {} at index {}, current threshold {}, but needed {}",
                    a, b, i, EQUALITY_THRESHOLD, diff
                );
                return false;
            }
        }
        true
    }
}
impl PartialEq for TestTensor<bool> {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        let mut counter = 0;
        let max_errors = (self.values.len() as f32 * EQUALITY_THRESHOLD) as usize;
        println!(
            "left trues:{}, right trues:{}, total vals: {}",
            self.values.iter().filter(|x| **x).count(),
            other.values.iter().filter(|x| **x).count(),
            self.values.len()
        );
        for (i, (a, b)) in self.values.iter().zip(other.values.iter()).enumerate() {
            if a != b {
                counter += 1;
                if counter > max_errors {
                    println!(
                        "TestTensor::eq: {} != {} at index {}, surpassed max errors: {}",
                        a, b, i, max_errors
                    );
                    return false;
                }
            }
        }
        true
    }
}

impl<T: std::fmt::Debug + Clone + PartialEq> std::fmt::Debug for TestTensor<T> {
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
