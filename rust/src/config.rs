

#[derive(Clone)]
pub struct TextEmbedConfig {
    pub chunk_size: Option<usize>,
    pub batch_size: Option<usize>,
}

impl Default for TextEmbedConfig {
    fn default() -> Self {
        Self {
            chunk_size: Some(256),
            batch_size: Some(32),
        }
    }
}

impl TextEmbedConfig {
    pub fn new(chunk_size: Option<usize>, batch_size: Option<usize>) -> Self {
        Self {
            chunk_size,
            batch_size,
        }
    }
}
