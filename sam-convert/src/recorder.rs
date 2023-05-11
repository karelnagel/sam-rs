use burn::{
    module::{Module, ModuleMapper, ParamId},
    tensor::{backend::Backend, Tensor},
};
use sam_rs::sam::Sam;
struct Mapper;
impl<B: Backend> ModuleMapper<B> for Mapper {
    fn map<const D: usize>(&mut self, _id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        println!("Mapping tensor...{}", _id);
        tensor
    }
}
pub fn load_module_from_python<B: Backend>(sam: Sam<B>) -> Sam<B> {
    let mut mapper = Mapper;
    sam.map(&mut mapper)
}
