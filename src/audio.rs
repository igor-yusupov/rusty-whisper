use hound::{Error, WavReader};
use ndarray_npy::NpzReader;
use rayon::prelude::*;
use rustfft::num_complex::ComplexFloat;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::fs::File;
use tract_onnx::tract_hir::tract_ndarray::{s, Array, Array2};

const N_FFT: usize = 400;
pub const N_FRAMES: usize = 3000;

fn pad_audio(audio: &Vec<f32>) -> Vec<f32> {
    let audio_len = audio.len();
    let pad_len = N_FFT / 2;

    let mut padded_audio = vec![0.0; audio_len + 2 * pad_len];

    for i in 0..audio_len {
        padded_audio[pad_len + i] = audio[i];
    }

    for i in 0..pad_len {
        padded_audio[i] = audio[pad_len - i];
        padded_audio[pad_len + audio_len + i] = audio[audio_len - 1 - i];
    }

    padded_audio
}

pub fn read_audio(file_path: &str) -> Result<Vec<f32>, Error> {
    let reader = WavReader::open(file_path)?;
    let mut audio_data = Vec::new();
    for sample in reader.into_samples::<i32>() {
        let sample_value = sample? as f32 / 32768.0;
        audio_data.push(sample_value);
    }

    Ok(audio_data)
}

pub fn log_mel_spectrogram(audio_data: Vec<f32>, filters: Array2<f32>) -> Array2<f32> {
    let stft = par_generate_stft(&pad_audio(&audio_data), 400, 160);
    // let stft = generate_stft(&pad_audio(&audio_data), 400, 160);
    let magnitudes: Array2<f32> = Array2::from_shape_fn((stft[0].len(), stft.len()), |(i, j)| {
        let element = stft[j][i].abs();
        element * element
    });
    let mel_spec = filters.dot(&magnitudes);
    let clipped_spec = mel_spec.map(|&x| if x < 1e-10 { 1e-10 } else { x });
    let log_spec = clipped_spec.map(|&x| x.log10());
    let max_value = *log_spec
        .iter()
        .max_by(|&x, &y| x.partial_cmp(y).unwrap())
        .unwrap();
    let log_spec = log_spec.map(|&x| {
        if x > max_value - 8.0 {
            x
        } else {
            max_value - 8.0
        }
    });
    let log_spec = log_spec.map(|&x| (x + 4.0) / 4.0);
    log_spec
}

pub fn pad_or_trim(mel: Array2<f32>, length: usize) -> Array2<f32> {
    if mel.shape()[1] > length {
        mel.slice(s![.., ..length]).to_owned()
    } else if mel.shape()[1] < length {
        let mut padded = Array::zeros((mel.shape()[0], length));
        padded.slice_mut(s![.., ..mel.shape()[1]]).assign(&mel);
        padded
    } else {
        mel
    }
}

fn generate_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (size - 1) as f32).cos()))
        .collect()
}

fn par_generate_stft(audio: &Vec<f32>, n_fft: usize, hop_length: usize) -> Vec<Vec<Complex<f32>>> {
    let window = generate_hann_window(n_fft);
    let fft = FftPlanner::new().plan_fft_forward(n_fft);

    let num_frames = (audio.len() - n_fft) / hop_length;
    let mut stft: Vec<Vec<Complex<f32>>> =
        vec![vec![Complex::new(0.0, 0.0); n_fft / 2 + 1]; num_frames];

    stft.par_iter_mut()
        .enumerate()
        .for_each(|(frame, frame_stft)| {
            let start = frame * hop_length;
            let end = start + n_fft;
            let frame_audio = &audio[start..end];

            let mut windowed_frame: Vec<Complex<f32>> = frame_audio
                .iter()
                .zip(window.iter())
                .map(|(&x, &w)| Complex::new(x * w, 0.0))
                .collect();

            fft.process(&mut windowed_frame);

            for (i, &val) in windowed_frame.iter().enumerate().take(n_fft / 2) {
                frame_stft[i] = val;
            }
        });

    stft
}

pub fn get_mel_filteres(mel_filteres_path: &str) -> Array2<f32> {
    let file = File::open(mel_filteres_path).expect("Failed to open file");
    let mut npz = NpzReader::new(file).expect("Failed to read NPZ file");
    let mel_filters: Array2<f32> = npz.by_index(0).unwrap();
    mel_filters
}
