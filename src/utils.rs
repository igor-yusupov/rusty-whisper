use rayon::prelude::*;
use tract_ndarray::{Array3, ArrayBase, Dim, OwnedRepr};
use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct BeamNode {
    pub tokens: ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>>,
    pub score: f32,
    pub kv_cache: KVCache,
}

pub fn get_top_indices(arr: Vec<f32>, n: usize) -> Vec<(usize, f32)> {
    let mut indexed_values: Vec<(usize, f32)> = arr
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect();
    indexed_values.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_n: Vec<(usize, f32)> = indexed_values.into_iter().take(n).collect();

    return top_n;
}

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
