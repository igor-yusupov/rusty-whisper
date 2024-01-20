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
    let result = whisper.recognize_from_audio("data/audio.wav","en");
    println!("{}", result);
}

```

The model works only with 16-bit WAV files, so make sure to convert your input before running the tool. For example, you can use ffmpeg like this:

```
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```
## Testrun on Macbook M1 
```
15 sec for init
224 sec for transcription
 The security of our nation is the most solemn responsibility of any government and the first priority of every Prime Minister. Today I want to take the opportunity to share my vision for an Australia that is stronger, safer and more resilient. More prepared to meet the challenges and threats of a less certain world. Almost 80 years ago on the 14th of March 1942 Prime Minister John Curtin gave a speech for broadcast on American radio. He began with this. On the great waters of the Pacific Ocean War now breathes its bloody steam. From the skies of the Pacific pours down a deadly hail. In the countless islands of the Pacific the tide of war flows madly. For you in America, for us in Australia it is flowing badly. Now Curtin was not one for doom saying or hyperbole. Truly they were the most fearful days our nation has known. Eight decades later Labour still looks to Curtin. Not just to salute his strength of character or his sacrifice but because Curtin's famous 1941 declaration that Australia looked to America was deeper than a statement of wartime necessity. It was an assertion of Australia's right and indeed Australia's responsibility to act in our own interests to make our own alliances to decide our place in our region for ourselves. And through 80 years of change that principle of sovereignty has remained at the core of Labour's approach to our foreign policy and our defence policy.
```
Provided audio File Rusty.wav was used 
