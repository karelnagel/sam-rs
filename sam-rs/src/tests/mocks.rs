use tch::nn;

use crate::tests::helpers::random_tensor;

trait Mock {
    fn mock() -> Self;
}
impl Mock for nn::Linear {
    fn mock() -> Self {
        Self {
            ws: random_tensor(&[256, 256], 1),
            bs: Some(random_tensor(&[256], 3)),
        }
    }
}
