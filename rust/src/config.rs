#[derive(Clone)]
pub struct BertConfig {
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub chunk_size: Option<usize>,
    pub batch_size: Option<usize>,
}

impl BertConfig {
    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        chunk_size: Option<usize>,
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            model_id,
            revision,
            chunk_size,
            batch_size,
        }
    }
}

#[derive(Clone)]
pub struct ClipConfig {
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub batch_size: Option<usize>,
}

impl ClipConfig {

    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            model_id,
            revision,
            batch_size,
        }
    }
}

#[derive(Clone)]
pub struct CloudConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub chunk_size: Option<usize>,
}

#[derive(Clone)]
pub struct JinaConfig {
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub chunk_size: Option<usize>,
    pub batch_size: Option<usize>,
}

impl JinaConfig {

    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        chunk_size: Option<usize>,
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            model_id,
            revision,
            chunk_size,
            batch_size,
        }
    }
}

impl CloudConfig {

    pub fn new(
        provider: Option<String>,
        model: Option<String>,
        api_key: Option<String>,
        chunk_size: Option<usize>,
    ) -> Self {
        Self {
            provider,
            model,
            api_key,
            chunk_size,
        }
    }
}

#[derive(Clone)]
pub struct AudioDecoderConfig {
    pub decoder_model_id: Option<String>,
    pub decoder_revision: Option<String>,
    pub model_type: Option<String>,
    pub quantized: Option<bool>,
}

impl AudioDecoderConfig {

    pub fn new(
        decoder_model_id: Option<String>,
        decoder_revision: Option<String>,
        model_type: Option<String>,
        quantized: Option<bool>,
    ) -> Self {
        Self {
            decoder_model_id,
            decoder_revision,
            model_type,
            quantized,
        }
    }
}

#[derive(Clone, Default)]
pub struct EmbedConfig {
    pub bert: Option<BertConfig>,
    pub clip: Option<ClipConfig>,
    pub cloud: Option<CloudConfig>,
    pub jina: Option<JinaConfig>,
    pub audio_decoder: Option<AudioDecoderConfig>,
}

impl EmbedConfig {

    pub fn new(
        bert: Option<BertConfig>,
        clip: Option<ClipConfig>,
        cloud: Option<CloudConfig>,
        jina: Option<JinaConfig>,
        audio_decoder: Option<AudioDecoderConfig>,
    ) -> Self {
        Self {
            bert,
            clip,
            cloud,
            jina,
            audio_decoder,
        }
    }
}
