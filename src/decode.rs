use tract_ndarray::{concatenate, Array, Axis, Dim};
use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct GreedyDecoder {
    eot: u32,
}

impl GreedyDecoder {
    pub fn new(eot: u32) -> GreedyDecoder {
        GreedyDecoder { eot }
    }

    pub fn update(
        self,
        tokens: Array<i32, Dim<[usize; 2]>>,
        logits: Array<f32, Dim<[usize; 2]>>,
    ) -> (Array<i32, Dim<[usize; 2]>>, bool) {
        let (_, next_word) =
            logits
                .iter()
                .enumerate()
                .fold((f32::MIN, None), |(max_val, max_pos), (idx, &val)| {
                    if val > max_val {
                        (val, Some(idx))
                    } else {
                        (max_val, max_pos)
                    }
                });
        let next_word = next_word.unwrap();

        let completed: bool;

        if next_word == self.eot as usize {
            completed = true;
        } else {
            completed = false;
        }

        let next_word_array = Array::from_elem((1, 1), next_word as i32);
        let tokens = concatenate!(Axis(1), tokens, next_word_array);

        (tokens, completed)
    }
}
