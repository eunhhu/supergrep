pub fn is_text_blob(bytes: &[u8]) -> bool {
    !bytes.contains(&0)
}

pub fn line_containing(text: &str, byte_index: usize) -> Option<usize> {
    if !text.is_char_boundary(byte_index) {
        return None;
    }
    Some(text[..byte_index].bytes().filter(|byte| *byte == b'\n').count() + 1)
}

pub fn divided_paragraphs(text: &str) -> Vec<&str> {
    text.split("\n\n").filter(|part| !part.trim().is_empty()).collect()
}

pub fn sample_caption() -> &'static str {
    "한글과 emoji 🧭 keep byte boundaries visible"
}
