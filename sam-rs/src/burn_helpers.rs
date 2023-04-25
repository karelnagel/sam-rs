use burn::{
    module::{Module, Param},
    tensor::{
        backend::Backend, Bool, Data, ElementConversion, Float, Int, Shape, Tensor, TensorKind,
    },
};

#[derive(Debug, Module)]
pub struct ConvTranspose2d<B: Backend> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    idk: Param<Tensor<B, 2>>, //Just added to use generic B
}
impl<B: Backend> ConvTranspose2d<B> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            idk: Tensor::zeros([1, 1]).into(),
        }
    }
    pub fn forward<const D2: usize>(&self, x: Tensor<B, D2>) -> Tensor<B, D2> {
        unimplemented!()
    }
}

pub trait TensorSlice<const D: usize, E> {
    fn of_slice<E2: Into<E>>(slice: Vec<E2>, shape: [usize; D]) -> Self;
    fn to_slice(&self) -> (Vec<E>, [usize; D]);
}

impl<B: Backend, const D: usize> TensorSlice<D, f32> for Tensor<B, D> {
    fn of_slice<E2: Into<f32>>(slice: Vec<E2>, shape: [usize; D]) -> Self {
        let slice: Vec<f32> = slice.into_iter().map(|a| a.into()).collect();
        let data = Data::new(slice, Shape::new(shape));
        Tensor::from_data(data.convert())
    }
    fn to_slice(&self) -> (Vec<f32>, [usize; D]) {
        let data = self.to_data();
        let value: Vec<f32> = data.value.iter().map(|a| a.elem()).collect();
        let shape = data.shape.dims;
        (value, shape)
    }
}
impl<B: Backend, const D: usize> TensorSlice<D, i32> for Tensor<B, D, Int> {
    fn of_slice<E2: Into<i32>>(slice: Vec<E2>, shape: [usize; D]) -> Self {
        let slice: Vec<i32> = slice.into_iter().map(|a| a.into()).collect();
        let data = Data::new(slice, Shape::new(shape));
        Tensor::from_data(data.convert())
    }
    fn to_slice(&self) -> (Vec<i32>, [usize; D]) {
        let data: Data<<B as Backend>::IntElem, D> = self.to_data();
        let value: Vec<i32> = data.value.iter().map(|a| a.elem()).collect();
        let shape = data.shape.dims;
        (value, shape)
    }
}

impl<B: Backend, const D: usize> TensorSlice<D, bool> for Tensor<B, D, Bool> {
    fn of_slice<E2: Into<bool>>(slice: Vec<E2>, shape: [usize; D]) -> Self {
        let slice: Vec<bool> = slice.into_iter().map(|a| a.into()).collect();
        let data = Data::new(slice, Shape::new(shape));
        Tensor::from_data(data)
    }
    fn to_slice(&self) -> (Vec<bool>, [usize; D]) {
        let data = self.to_data();
        let value: Vec<bool> = data.value;
        let shape = data.shape.dims;
        (value, shape)
    }
}
