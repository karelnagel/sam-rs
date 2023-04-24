use burn::{
    module::{Module, Param},
    tensor::{
        backend::Backend, Bool, Data, ElementConversion, Float, Int, Shape, Tensor, TensorKind,
    },
};
pub trait TensorAddons<B: Backend, const D: usize, K: TensorKind<B> = Float> {
    fn permute<const D2: usize, S: Into<Shape<D2>>>(&self, dims: S) -> Tensor<B, D2, K>;
    fn unbind<const D2: usize>(&self, dim: usize) -> Vec<Tensor<B, D2, K>>;
    fn cumsum(&self, dim: usize) -> Tensor<B, D, K>;
    fn stack<const D2: usize>(tensors: Vec<Tensor<B, D, K>>, dim: usize) -> Tensor<B, D2, K>;
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Tensor<B, D, K>;
    fn upsample_linear1d<const D2: usize>(
        &self,
        output_size: &[usize],
        align_corners: bool,
        scales: impl Into<Option<f64>>,
    ) -> Tensor<B, D2>;
    fn squeeze_dim<const D2: usize>(&self, dim: usize) -> Tensor<B, D2, K>;
    fn slice(
        &self,
        dim: usize,
        start: impl Into<Option<usize>>,
        end: impl Into<Option<usize>>,
        step: usize,
    ) -> Tensor<B, D, K>;
    fn pad(&self, pad: &[usize], mode: &str, value: impl Into<Option<f64>>) -> Tensor<B, D, K>;
    fn expand(&self, size: Vec<usize>, implicit: bool) -> Tensor<B, D, K>;
    fn repeat_interleave_self_int(
        &self,
        repeats: usize,
        dim: impl Into<Option<usize>>,
        output_size: impl Into<Option<usize>>,
    ) -> Tensor<B, D, K>;
    fn where_self(&self, condition: Tensor<B, D, Bool>, other: Tensor<B, D, K>) -> Tensor<B, D, K>;
    fn expand_as(&self, other: Tensor<B, D, K>) -> Tensor<B, D, K>;
    fn copy_(&mut self, src: Tensor<B, D, K>);
    fn constant_pad_nd<const D2: usize>(&self, pad: Vec<usize>) -> Tensor<B, D2, K>;
    fn upsample_bilinear2d<const D2: usize>(
        &self,
        output_size: Vec<usize>,
        align_corners: bool,
        scales_h: impl Into<Option<f64>>,
        scales_w: impl Into<Option<f64>>,
    ) -> Tensor<B, D2, K>;
    fn select<const D2: usize>(&self, dim: i64, index: i64) -> Tensor<B, D2, K>;
    fn flip(&self, dims: Vec<usize>) -> Tensor<B, D, K>;
}

impl<B: Backend, const D: usize, K: TensorKind<B>> TensorAddons<B, D, K> for Tensor<B, D, K> {
    fn permute<const D2: usize, S: Into<Shape<D2>>>(&self, dims: S) -> Tensor<B, D2, K> {
        unimplemented!()
    }
    fn unbind<const D2: usize>(&self, dim: usize) -> Vec<Tensor<B, D2, K>> {
        unimplemented!()
    }
    fn cumsum(&self, dim: usize) -> Self {
        unimplemented!()
    }
    fn stack<const D2: usize>(tensors: Vec<Tensor<B, D, K>>, dim: usize) -> Tensor<B, D2, K> {
        unimplemented!()
    }
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Tensor<B, D, K> {
        unimplemented!()
    }
    fn upsample_linear1d<const D2: usize>(
        &self,
        output_size: &[usize],
        align_corners: bool,
        scales: impl Into<Option<f64>>,
    ) -> Tensor<B, D2> {
        unimplemented!()
    }
    fn squeeze_dim<const D2: usize>(&self, dim: usize) -> Tensor<B, D2, K> {
        unimplemented!()
    }
    fn slice(
        &self,
        dim: usize,
        start: impl Into<Option<usize>>,
        end: impl Into<Option<usize>>,
        step: usize,
    ) -> Self {
        unimplemented!()
    }
    fn pad(&self, pad: &[usize], mode: &str, value: impl Into<Option<f64>>) -> Self {
        unimplemented!()
    }
    fn expand(&self, size: Vec<usize>, implicit: bool) -> Self {
        unimplemented!()
    }
    fn repeat_interleave_self_int(
        &self,
        repeats: usize,
        dim: impl Into<Option<usize>>,
        output_size: impl Into<Option<usize>>,
    ) -> Tensor<B, D, K> {
        unimplemented!()
    }
    fn where_self(&self, condition: Tensor<B, D, Bool>, other: Tensor<B, D, K>) -> Self {
        unimplemented!()
    }
    fn expand_as(&self, other: Tensor<B, D, K>) -> Self {
        unimplemented!()
    }
    fn copy_(&mut self, src: Tensor<B, D, K>) {
        unimplemented!()
    }
    fn constant_pad_nd<const D2: usize>(&self, pad: Vec<usize>) -> Tensor<B, D2, K> {
        unimplemented!()
    }
    fn upsample_bilinear2d<const D2: usize>(
        &self,
        output_size: Vec<usize>,
        align_corners: bool,
        scales_h: impl Into<Option<f64>>,
        scales_w: impl Into<Option<f64>>,
    ) -> Tensor<B, D2, K> {
        unimplemented!()
    }
    fn select<const D2: usize>(&self, dim: i64, index: i64) -> Tensor<B, D2, K> {
        unimplemented!()
    }
    fn flip(&self, dims: Vec<usize>) -> Self {
        unimplemented!()
    }
}

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
