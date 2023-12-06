# rusty-whisper

Rust implementation of Whisper. More information about model: https://github.com/openai/whisper

This crate is based on [tract](https://github.com/sonos/tract).

## Quick start

Download [weights](https://www.dropbox.com/scl/fi/obq73jdswc9yfu2f8ctwo/weights.zip?rlkey=iuofo1dbf2xo6hiu9io6ovh5i&dl=1) and [example audio file](https://www.dropbox.com/scl/fi/8yzo8y2ptxoy0rfuon9bu/audio.wav?rlkey=dorb43edb48bqpx5cgrtckxlk&dl=1) and run simple inference code:

```
use rusty_whisper::Whisper;

fn main() {
    let whisper = Whisper::new(
        "weights/encoder.onnx",
        "weights/decoder.onnx",
        "weights/multilingual.tiktoken",
        "weights/positional_embedding.npz",
        "weights/mel_filters.npz",
    );
    let result = whisper.recognize_from_audio("data/audio.wav");
    println!("{}", result);
}

```

The model works only with 16-bit WAV files, so make sure to convert your input before running the tool. For example, you can use ffmpeg like this:

```
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```
