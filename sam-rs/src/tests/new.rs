use burn::tensor::{backend::Backend, Int, Tensor};
use pyo3::PyAny;

use crate::burn_helpers::TensorSlice;

use super::helpers::TEST_ALMOST_THRESHOLD;

#[derive(PartialEq)]
pub struct TestTensor2 {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

impl std::fmt::Debug for TestTensor2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = &self.values.len();
        if self.values.len() <= 10 {
            return f
                .debug_struct("TestTensor")
                .field("size", &self.shape)
                .field("values", &self.values)
                .finish();
        }
        f.debug_struct("TestTensor")
            .field("size", &self.shape)
            .field("start", &self.values[0..5].to_vec())
            .field("end", &self.values[len - 6..len - 1].to_vec())
            .finish()
    }
}

fn difference(a: f32, b: f32) -> f32 {
    (a - b).abs() / a.abs().max(b.abs())
}
impl TestTensor2 {
    pub fn almost_equal<T: Into<Self>, K: Into<Option<f32>>>(&self, other: T, threshold: K) {
        let threshold = threshold.into().unwrap_or(TEST_ALMOST_THRESHOLD);
        let other = other.into();
        if self.shape != other.shape {
            panic!("TestTensor sizes don't match");
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
            let diff = difference(*a, *b);
            if diff <= threshold {
                almost += 1;
                continue;
            };
            max_diff = max_diff.max(diff);
            failed += 1;
        }
        let total = self.values.len();

        match failed {
            0 => {}
            _ => {
                println!(
                    "TestTensor::eq: exact: {}, almost: {}, failed: {}, total: {}! Max threshold: {}, current: {}",
                    exact, almost, failed, total,max_diff, threshold
                );
                println!("left: {:?}", self);
                println!("right: {:?}", other);
                panic!("almost equal failed");
            }
        }
    }
}

impl<B: Backend, const D: usize> From<Tensor<B, D>> for TestTensor2 {
    fn from(tensor: Tensor<B, D>) -> Self {
        let (values, shape) = tensor.to_slice();
        TestTensor2 {
            shape: shape.to_vec(),
            values,
        }
    }
}
impl<B: Backend, const D: usize> From<Tensor<B, D, Int>> for TestTensor2 {
    fn from(tensor: Tensor<B, D, Int>) -> Self {
        let (values, shape) = tensor.to_slice();
        let values = values.iter().map(|x| *x as f32).collect();
        TestTensor2 {
            shape: shape.to_vec(),
            values,
        }
    }
}

impl From<&PyAny> for TestTensor2 {
    fn from(tensor: &PyAny) -> Self {
        let shape = tensor
            .getattr("shape")
            .unwrap()
            .extract::<Vec<usize>>()
            .unwrap();
        let values = tensor
            .getattr("flatten")
            .unwrap()
            .call0()
            .unwrap()
            .extract::<Vec<f32>>()
            .unwrap();
        Self { values, shape }
    }
}
