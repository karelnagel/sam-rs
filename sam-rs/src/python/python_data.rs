use burn::tensor::{backend::Backend, BasicOps, Element, ElementConversion, Tensor, TensorKind};
use pyo3::{types::PyTuple, FromPyObject, PyAny, PyErr, PyResult, Python};

use crate::{
    burn_helpers::TensorHelpers, sam_predictor::Size, tests::helpers::TEST_ALMOST_THRESHOLD,
};

pub trait PythonDataKind: std::fmt::Debug + PartialEq + Clone + Element + Sized + Copy {}
impl PythonDataKind for f32 {}
impl PythonDataKind for i64 {}
pub fn random_python_tensor<const D: usize>(py: Python, shape: [usize; D]) -> PyResult<&PyAny> {
    let torch = py.import("torch")?;
    let input = torch.call_method1("randn", (shape,))?;
    Ok(input)
}
pub fn random_python_tensor_int<const D: usize>(py: Python, shape: [usize; D]) -> PyResult<&PyAny> {
    let tensor = random_python_tensor(py, shape)?;
    let int = py.import("torch")?.getattr("int")?;
    let tensor = tensor.call_method1("type", (int,))?;
    Ok(tensor)
}
#[derive(PartialEq, Clone)]
pub struct PythonData<const D: usize, T: PythonDataKind = f32> {
    pub slice: Vec<T>,
    pub shape: [usize; D],
}

impl<T: PythonDataKind, const D: usize> std::fmt::Debug for PythonData<D, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.slice.len();
        if len <= 10 {
            return f
                .debug_struct("TestTensor")
                .field("shape", &self.shape)
                .field("values", &self.slice)
                .finish();
        }
        f.debug_struct("TestTensor")
            .field("shape", &self.shape)
            .field("start", &self.slice[0..5].to_vec())
            .field("end", &self.slice[len - 6..len - 1].to_vec())
            .finish()
    }
}
impl<const D: usize, T: PythonDataKind> PythonData<D, T> {
    pub fn new(slice: Vec<T>, shape: [usize; D]) -> Self {
        Self { slice, shape }
    }
    pub fn equal<I: Into<Self>>(&self, other: I) {
        let other = other.into();
        assert_eq!(self, &other, "PythonData::eq failed");
    }

    pub fn almost_equal<I: Into<Self>, X: Into<Option<f32>>>(&self, output: I, threshold: X) {
        let other: Self = output.into();
        let threshold = threshold.into().unwrap_or(TEST_ALMOST_THRESHOLD);
        if self.shape != other.shape {
            panic!("TestTensor sizes don't match");
        }
        let mut exact = 0;
        let mut almost = 0;
        let mut failed = 0;
        let mut max_diff: f32 = 0.0;
        for (a, b) in self.slice.iter().zip(other.slice.iter()) {
            let a = a.to_f32().unwrap();
            let b = b.to_f32().unwrap();
            if a == b {
                exact += 1;
                continue;
            }
            let diff = (a - b).abs() / a.abs().max(b.abs());
            if diff <= threshold {
                almost += 1;
                continue;
            };
            max_diff = max_diff.max(diff);
            failed += 1;
        }
        let total = self.slice.len();

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

impl<'a, const D: usize, T: PythonDataKind> TryFrom<&'a PyAny> for PythonData<D, T>
where
    Vec<T>: FromPyObject<'a>,
{
    type Error = PyErr;
    fn try_from(data: &'a PyAny) -> PyResult<Self> {
        let slice = data
            .getattr("flatten")?
            .call0()?
            .getattr("tolist")?
            .call0()?
            .extract::<Vec<T>>()?;
        let shape = data.getattr("shape")?.extract::<Vec<usize>>()?;
        assert_eq!(D, shape.len(), "Shape length doesn't match");
        let shape = shape.try_into().unwrap();
        Ok(PythonData::new(slice.to_vec(), shape))
    }
}

impl<B: Backend, const D: usize, T: PythonDataKind, K: TensorKind<B> + BasicOps<B>>
    From<PythonData<D, T>> for Tensor<B, D, K>
where
    <K as BasicOps<B>>::Elem: ElementConversion,
{
    fn from(data: PythonData<D, T>) -> Self {
        let slice = data.slice;
        let shape = data.shape;
        Tensor::of_slice(slice, shape)
    }
}

impl<B: Backend, const D: usize, T: PythonDataKind, K: TensorKind<B> + BasicOps<B>>
    From<Tensor<B, D, K>> for PythonData<D, T>
where
    <K as BasicOps<B>>::Elem: ElementConversion,
{
    fn from(data: Tensor<B, D, K>) -> Self {
        let (slice, shape) = data.to_slice();
        PythonData::new(slice, shape)
    }
}

impl TryFrom<&PyAny> for Size {
    type Error = PyErr;
    fn try_from(data: &PyAny) -> PyResult<Self> {
        let tuple = data.downcast::<PyTuple>()?;
        Ok(Size(
            tuple.get_item(0)?.extract()?,
            tuple.get_item(1)?.extract()?,
        ))
    }
}
