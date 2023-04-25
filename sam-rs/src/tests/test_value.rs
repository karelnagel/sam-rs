use burn::tensor::{backend::Backend, Bool, Tensor};
use serde::{Deserialize, Serialize};

use crate::{
    burn_helpers::TensorSlice,
    modeling::common::activation::Activation,
    sam_predictor::{ImageFormat, Size},
};
#[derive(Deserialize, PartialEq, Serialize)]
pub struct TestTensor<T> {
    size: Vec<usize>,
    values: Vec<T>,
}

impl<T: std::fmt::Debug + Clone> std::fmt::Debug for TestTensor<T> {
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
