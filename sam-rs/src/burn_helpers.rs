use burn::tensor::{
    backend::Backend, BasicOps, Bool, Data, ElementConversion, Int, Shape, Tensor, TensorKind,
};

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
pub trait TensorHelpers<B: Backend, const D: usize> {
    fn calc_dims<const D2: usize>(&self, dims: [usize; D2]) -> [usize; D2];
    fn reshape_max<const D2: usize>(&self, dims: [usize; D2]) -> Tensor<B, D2>;
}
impl<B: Backend, const D: usize> TensorHelpers<B, D> for Tensor<B, D> {
    fn calc_dims<const D2: usize>(&self, dims: [usize; D2]) -> [usize; D2] {
        let max_count = dims.iter().filter(|&&x| x == usize::MAX).count();
        assert!(
            max_count <= 1,
            "There mustca be exactly one usize::MAX in the dims array"
        );
        if max_count == 0 {
            return dims;
        }
        let elems = self.dims().iter().fold(1, |acc, &x| acc * x)
            / dims
                .iter()
                .filter(|x| **x != usize::MAX)
                .fold(1, |acc, &x| acc * x);
        dims.map(|x| if x == usize::MAX { elems } else { x })
    }
    fn reshape_max<const D2: usize>(&self, dims: [usize; D2]) -> Tensor<B, D2> {
        self.clone().reshape(self.calc_dims(dims))
    }
}

pub trait TensorHelpers2<B: Backend, const D: usize, K: TensorKind<B>> {
    fn unsqueeze_end<const D2: usize>(self) -> Tensor<B, D2, K>;
}
impl<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>> TensorHelpers2<B, D, K>
    for Tensor<B, D, K>
{
    fn unsqueeze_end<const D2: usize>(self) -> Tensor<B, D2, K> {
        let tensor = self.unsqueeze();
        let mut dims = [0; D2];
        let diff = D2 - D;
        for i in 0..D2 {
            dims[i] = match i < D {
                true => i + diff,
                false => i - D,
            }
        }
        let tensor = tensor.permute(dims);
        tensor
    }
}
