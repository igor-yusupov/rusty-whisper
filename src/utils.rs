use tract_ndarray::{Array3, Dim};
use tract_onnx::prelude::*;

#[derive(Debug)]
pub struct Options {
    pub eot_token: usize,
    pub sot_prev: usize,
    pub n_ctx: usize,
}

impl Options {
    pub fn new() -> Options {
        Options {
            eot_token: 50257,
            sot_prev: 50361,
            n_ctx: 448,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub k1: Tensor,
    pub k2: Tensor,
    pub k3: Tensor,
    pub k4: Tensor,
    pub k5: Tensor,
    pub k6: Tensor,
    pub v1: Tensor,
    pub v2: Tensor,
    pub v3: Tensor,
    pub v4: Tensor,
    pub v5: Tensor,
    pub v6: Tensor,
}

impl KVCache {
    pub fn default() -> KVCache {
        let shape = Dim([1, 0, 512]);
        let value: Array3<f32> = Array3::zeros(shape);

        KVCache {
            k1: value.clone().into_tensor(),
            k2: value.clone().into_tensor(),
            k3: value.clone().into_tensor(),
            k4: value.clone().into_tensor(),
            k5: value.clone().into_tensor(),
            k6: value.clone().into_tensor(),
            v1: value.clone().into_tensor(),
            v2: value.clone().into_tensor(),
            v3: value.clone().into_tensor(),
            v4: value.clone().into_tensor(),
            v5: value.clone().into_tensor(),
            v6: value.clone().into_tensor(),
        }
    }
}
