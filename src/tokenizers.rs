use base64::{engine::general_purpose, Engine as _};
use rustc_hash::FxHashMap as HashMap;
use std::fs::File;
use std::io::Read;
use tiktoken_rs::CoreBPE;

#[derive(Debug)]
pub struct Tokenizer {
    bpe: CoreBPE,
    pub lang2token: HashMap<String, usize>,
}

impl Tokenizer {
    pub fn new(vocab_path: &str) -> Tokenizer {
        let mut file = File::open(vocab_path).expect("Не удалось открыть файл");

        let langs = vec![
            ("en", "english"),
            ("zh", "chinese"),
            ("de", "german"),
            ("es", "spanish"),
            ("ru", "russian"),
            ("ko", "korean"),
            ("fr", "french"),
            ("ja", "japanese"),
            ("pt", "portuguese"),
            ("tr", "turkish"),
            ("pl", "polish"),
            ("ca", "catalan"),
            ("nl", "dutch"),
            ("ar", "arabic"),
            ("sv", "swedish"),
            ("it", "italian"),
            ("id", "indonesian"),
            ("hi", "hindi"),
            ("fi", "finnish"),
            ("vi", "vietnamese"),
            ("iw", "hebrew"),
            ("uk", "ukrainian"),
            ("el", "greek"),
            ("ms", "malay"),
            ("cs", "czech"),
            ("ro", "romanian"),
            ("da", "danish"),
            ("hu", "hungarian"),
            ("ta", "tamil"),
            ("no", "norwegian"),
            ("th", "thai"),
            ("ur", "urdu"),
            ("hr", "croatian"),
            ("bg", "bulgarian"),
            ("lt", "lithuanian"),
            ("la", "latin"),
            ("mi", "maori"),
            ("ml", "malayalam"),
            ("cy", "welsh"),
            ("sk", "slovak"),
            ("te", "telugu"),
            ("fa", "persian"),
            ("lv", "latvian"),
            ("bn", "bengali"),
            ("sr", "serbian"),
            ("az", "azerbaijani"),
            ("sl", "slovenian"),
            ("kn", "kannada"),
            ("et", "estonian"),
            ("mk", "macedonian"),
            ("br", "breton"),
            ("eu", "basque"),
            ("is", "icelandic"),
            ("hy", "armenian"),
            ("ne", "nepali"),
            ("mn", "mongolian"),
            ("bs", "bosnian"),
            ("kk", "kazakh"),
            ("sq", "albanian"),
            ("sw", "swahili"),
            ("gl", "galician"),
            ("mr", "marathi"),
            ("pa", "punjabi"),
            ("si", "sinhala"),
            ("km", "khmer"),
            ("sn", "shona"),
            ("yo", "yoruba"),
            ("so", "somali"),
            ("af", "afrikaans"),
            ("oc", "occitan"),
            ("ka", "georgian"),
            ("be", "belarusian"),
            ("tg", "tajik"),
            ("sd", "sindhi"),
            ("gu", "gujarati"),
            ("am", "amharic"),
            ("yi", "yiddish"),
            ("lo", "lao"),
            ("uz", "uzbek"),
            ("fo", "faroese"),
            ("ht", "haitian creole"),
            ("ps", "pashto"),
            ("tk", "turkmen"),
            ("nn", "nynorsk"),
            ("mt", "maltese"),
            ("sa", "sanskrit"),
            ("lb", "luxembourgish"),
            ("my", "myanmar"),
            ("bo", "tibetan"),
            ("tl", "tagalog"),
            ("mg", "malagasy"),
            ("as", "assamese"),
            ("tt", "tatar"),
            ("haw", "hawaiian"),
            ("ln", "lingala"),
            ("ha", "hausa"),
            ("ba", "bashkir"),
            ("jw", "javanese"),
            ("su", "sundanese"),
        ];
        let languages: HashMap<&str, &str> = langs.iter().cloned().collect();

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Не удалось прочитать файл");

        let mut encoder = HashMap::default();

        for line in contents.lines() {
            let mut parts = line.split(' ');
            let word: String = parts.next().unwrap().parse().unwrap();
            let rank: usize = parts.next().unwrap().parse().unwrap();

            let token = &general_purpose::STANDARD.decode(word);

            match token {
                Ok(value) => {
                    encoder.insert(value.clone(), rank);
                }
                Err(_) => {
                    let token: Vec<u8> = vec![];
                    encoder.insert(token.clone(), rank);
                }
            }
        }

        let mut special_tokens = HashMap::default();
        let n_vocab: usize = encoder.len();

        let mut specials = vec![
            "<|endoftext|>".to_string(),
            "<|startoftranscript|>".to_string(),
        ];

        for lang in languages.keys() {
            specials.push(format!("<|{}|>", lang));
        }
        specials.extend(vec![
            "<|translate|>".to_string(),
            "<|transcribe|>".to_string(),
            "<|startoflm|>".to_string(),
            "<|startofprev|>".to_string(),
            "<|nospeech|>".to_string(),
            "<|notimestamps|>".to_string(),
        ]);
        for i in 0..1501 {
            let formatted = format!("<|{:.2}|>", i as f32 * 0.02);
            specials.push(formatted);
        }

        for (index, value) in specials.iter().enumerate() {
            special_tokens.insert(value.into(), index + n_vocab + 1);
        }

        let bpe = CoreBPE::new(
            encoder,
            special_tokens,
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        )
        .unwrap();

        let lang2token: HashMap<String, usize> = langs
            .iter()
            .enumerate()
            .map(|(index, pairs)| {
                (
                    String::from(pairs.0),
                    bpe.encode_with_special_tokens("<|startoftranscript|>")[0] + index,
                )
            })
            .collect();
        Tokenizer { bpe, lang2token }
    }

    // pub fn encode(&self, text: &str) -> Vec<usize> {
    //     self.bpe.encode_with_special_tokens(text)
    // }

    pub fn decode(&self, tokens: Vec<usize>) -> String {
        self.bpe.decode(tokens).unwrap()
    }
}
