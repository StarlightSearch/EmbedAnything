/// Data struct about the available models
#[derive(Debug, Clone)]
pub struct ModelInfo<T> {
    pub model: T,
    pub dim: usize,
    pub description: String,
    pub hf_model_id: String,
    pub model_code: String,
    pub model_file: String,
}
