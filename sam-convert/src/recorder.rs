use std::collections::HashMap;

use burn::{module::Module, tensor::backend::Backend};
use pyo3::{types::PyTuple, PyAny, PyResult};
use regex::Regex;
use sam_rs::sam::Sam;

use crate::update_tensor::update_tensor;
fn key_replacer(key: String) -> String {
    // Replacing "neck.1.", "output_upscaling.1.", "mask_downscaling.1." with neck1.
    let re = Regex::new(r"(neck|output_upscaling|mask_downscaling)\.(\d+)\.").unwrap();
    let key = re.replace_all(&key, "$1$2.").to_string();

    // Replacing all norm1.weight, norm2.weight with norm1.gamma, norm2.gamma
    let re = Regex::new(r"norm(\d+)\.weight").unwrap();
    let key = re.replace_all(&key, "norm$1.gamma").to_string();

    // Replacing all norm1.bias with norm1.beta
    let re = Regex::new(r"norm(\d+)\.bias").unwrap();
    let key = re.replace_all(&key, "norm$1.beta").to_string();

    // Replacing all norm_final_attn.weight with norm_final_attn.gamma
    let re = Regex::new(r"norm_final_attn\.weight").unwrap();
    let key = re.replace_all(&key, "norm_final_attn.gamma").to_string();

    // Replacing all norm_final_attn.bias with norm_final_attn.beta
    let re = Regex::new(r"norm_final_attn\.bias").unwrap();
    let key = re.replace_all(&key, "norm_final_attn.beta").to_string();

    // Replacing all .1. with [1].
    let re = Regex::new(r"\.(\d+)\.").unwrap();
    let key = re.replace_all(&key, "[$1].").to_string();

    key
}
pub fn get_python_map<'a>(sam: &'a PyAny) -> PyResult<HashMap<String, &'a PyAny>> {
    let mut map = HashMap::new();
    let params = sam.call_method0("named_parameters")?;
    let params = params.iter()?;
    for param in params {
        let param = param?.downcast::<PyTuple>()?;
        let key = param.get_item(0)?.extract::<String>()?;

        let key = key_replacer(key);

        let value = param.get_item(1)?;
        map.insert(key, value);
    }

    Ok(map)
}

pub fn load_sam<B: Backend>(sam: Sam<B>, values: HashMap<String, &PyAny>) -> Sam<B> {
    let mut record = sam.clone().into_record();
    for (key, value) in values.iter() {
        update_tensor(&mut record, key, value);
    }
    sam.load_record(record)
}
