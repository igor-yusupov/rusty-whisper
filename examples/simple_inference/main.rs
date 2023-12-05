use rusty_whisper::Whisper;

fn main() {
    let whisper = Whisper::new(
        "weights/encoder.onnx",
        "weights/decoder.onnx",
        "weights/multilingual.tiktoken",
        "weights/positional_embedding.npz",
        "weights/mel_filters.npz",
    );
    let result = whisper.recognize_from_audio("data/audio.wav", "en");
    println!("{}", result);
}
