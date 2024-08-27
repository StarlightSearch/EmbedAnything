use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct JinaConfig {
    pub inner: embed_anything::config::JinaConfig,
}

#[pymethods]
impl JinaConfig {
    #[new]
    #[pyo3(signature = (model_id=None, revision=None, chunk_size=None, batch_size=None))]
    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        chunk_size: Option<usize>,
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            inner: embed_anything::config::JinaConfig::new(
                model_id, revision, chunk_size, batch_size,
            ),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ClipConfig {
    pub inner: embed_anything::config::ClipConfig,
}

#[pymethods]
impl ClipConfig {
    #[new]
    #[pyo3(signature = (model_id=None, revision=None, batch_size=None))]
    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            inner: embed_anything::config::ClipConfig::new(model_id, revision, batch_size),
        }
    }

    #[getter]
    pub fn model_id(&self) -> Option<String> {
        self.inner.model_id.clone()
    }

    #[getter]
    pub fn revision(&self) -> Option<String> {
        self.inner.revision.clone()
    }

    #[getter]
    pub fn batch_size(&self) -> Option<usize> {
        self.inner.batch_size
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CloudConfig {
    pub inner: embed_anything::config::CloudConfig,
}

#[pymethods]
impl CloudConfig {
    #[new]
    #[pyo3(signature = (provider = "OpenAI".to_string(), model=None, api_key=None, chunk_size=None))]
    pub fn new(
        provider: Option<String>,
        model: Option<String>,
        api_key: Option<String>,
        chunk_size: Option<usize>,
    ) -> Self {
        Self {
            inner: embed_anything::config::CloudConfig::new(provider, model, api_key, chunk_size),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct AudioDecoderConfig {
    pub inner: embed_anything::config::AudioDecoderConfig,
}

#[pymethods]
impl AudioDecoderConfig {
    #[new]
    #[pyo3(signature = (decoder_model_id=None, decoder_revision=None, model_type=None, quantized=None))]
    pub fn new(
        decoder_model_id: Option<String>,
        decoder_revision: Option<String>,
        model_type: Option<String>,
        quantized: Option<bool>,
    ) -> Self {
        Self {
            inner: embed_anything::config::AudioDecoderConfig::new(
                decoder_model_id,
                decoder_revision,
                model_type,
                quantized,
            ),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BertConfig {
    pub inner: embed_anything::config::BertConfig,
}

#[pymethods]
impl BertConfig {
    #[new]
    #[pyo3(signature = (model_id=None, revision=None, chunk_size=None, batch_size=None))]
    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        chunk_size: Option<usize>,
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            inner: embed_anything::config::BertConfig::new(
                model_id, revision, chunk_size, batch_size,
            ),
        }
    }

    #[getter]
    pub fn model_id(&self) -> Option<String> {
        self.inner.model_id.clone()
    }

    #[getter]
    pub fn revision(&self) -> Option<String> {
        self.inner.revision.clone()
    }

    #[getter]
    pub fn chunk_size(&self) -> Option<usize> {
        self.inner.chunk_size
    }
}

#[pyclass]
#[derive(Clone)]
pub struct EmbedConfig {
    pub inner: embed_anything::config::EmbedConfig,
}

#[pymethods]
impl EmbedConfig {
    #[new]
    #[pyo3(signature = (bert=None, clip = None, cloud=None, jina=None, audio_decoder=None))]
    pub fn new(
        bert: Option<BertConfig>,
        clip: Option<ClipConfig>,
        cloud: Option<CloudConfig>,
        jina: Option<JinaConfig>,
        audio_decoder: Option<AudioDecoderConfig>,
    ) -> Self {
        Self {
            inner: embed_anything::config::EmbedConfig {
                bert: bert.map(|b| b.inner),
                clip: clip.map(|c| c.inner),
                cloud: cloud.map(|c| c.inner),
                jina: jina.map(|c| c.inner),
                audio_decoder: audio_decoder.map(|c| c.inner),
            },
        }
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct TextEmbedConfig {
    pub inner: embed_anything::config::TextEmbedConfig,
}

#[pymethods]
impl TextEmbedConfig {
    #[new]
    #[pyo3(signature = (chunk_size=None, batch_size=None))]
    pub fn new(chunk_size: Option<usize>, batch_size: Option<usize>) -> Self {
        Self {
            inner: embed_anything::config::TextEmbedConfig::new(chunk_size, batch_size),
        }
    }

    #[getter]
    pub fn chunk_size(&self) -> Option<usize> {
        self.inner.chunk_size
    }

    #[getter]
    pub fn batch_size(&self) -> Option<usize> {
        self.inner.batch_size
    }
}
