use md5::{Digest, Md5};
use tch::Tensor;

pub fn hash(value: String) -> String {
    let mut hasher = Md5::new();
    hasher.update(value.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

pub fn hash_tensor(tensor: Tensor) -> String {
    let flattened = tensor.flatten(0, -1);
    let value = format!("{:?}", flattened);
    let hash = hash(value);
    println!("hash: {:?}", hash);
    return hash;
}
