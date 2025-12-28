pub mod config;
pub mod models;
pub mod s3_client;
use embed_anything::embeddings::embed::{TextEmbedder, VisionEmbedder};
use embed_anything::{
    self,
    config::TextEmbedConfig,
    emb_audio,
    embeddings::embed::{Embedder, EmbeddingResult},
    file_processor::audio::audio_processor,
    FileLoadingError,
};
use models::colbert::ColbertModel;
use models::colpali::ColpaliModel;
use models::reranker::{DocumentRank, Dtype, Reranker, RerankerResult};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyValueError},
    prelude::*,
    types::PyList,
};
use std::fmt;
use std::str::FromStr;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use strum::EnumString;
use tokio::runtime::Builder;

#[pyclass]
pub struct EmbedData {
    pub inner: embed_anything::embeddings::embed::EmbedData,
}

#[pymethods]
impl EmbedData {
    #[getter(embedding)]
    fn embedding(&self) -> Py<PyList> {
        Python::with_gil(|py| {
            let embedding = self.inner.embedding.clone();
            match embedding {
                EmbeddingResult::DenseVector(x) => PyList::new(py, x).unwrap().into(),
                EmbeddingResult::MultiVector(x) => {
                    PyList::new(py, x.iter().map(|inner| PyList::new(py, inner).unwrap()))
                        .unwrap()
                        .into()
                }
            }
        })
    }

    #[getter(text)]
    fn text(&self) -> Option<String> {
        self.inner.text.clone()
    }

    #[getter(metadata)]
    fn metadata(&self) -> Option<HashMap<String, String>> {
        self.inner.metadata.clone()
    }

    #[setter(text)]
    fn set_text(&mut self, text: Option<String>) {
        self.inner.text = text;
    }

    #[setter(metadata)]
    fn set_metadata(&mut self, metadata: Option<HashMap<String, String>>) {
        self.inner.metadata = metadata;
    }

    fn __str__(&self) -> String {
        format!(
            "EmbedData(embedding: {:?}, text: {:?}, metadata: {:?})",
            self.inner.embedding,
            self.inner.text,
            self.inner.metadata.clone()
        )
    }

    fn __repr__(&self) -> String {
        "<class 'EmbedData'>".to_string()
    }
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Eq, Hash, EnumString, Debug)]
#[strum(serialize_all = "snake_case")]
pub enum WhichModel {
    OpenAI,
    Cohere,
    CohereVision,
    Bert,
    Model2Vec,
    Qwen3,
    SparseBert,
    ColBert,
    Clip,
    Jina,
    ModernBert,
    Colpali,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, EnumString)]
pub enum ONNXModel {
    AllMiniLML6V2,
    AllMiniLML6V2Q,
    AllMiniLML12V2,
    AllMiniLML12V2Q,
    ModernBERTBase,
    ModernBERTLarge,
    BGEBaseENV15,
    BGEBaseENV15Q,
    BGELargeENV15,
    BGELargeENV15Q,
    BGESmallENV15,
    BGESmallENV15Q,
    NomicEmbedTextV1,
    NomicEmbedTextV15,
    NomicEmbedTextV15Q,
    ParaphraseMLMiniLML12V2,
    ParaphraseMLMiniLML12V2Q,
    ParaphraseMLMpnetBaseV2,
    BGESmallZHV15,
    MultilingualE5Small,
    MultilingualE5Base,
    MultilingualE5Large,
    MxbaiEmbedLargeV1,
    MxbaiEmbedLargeV1Q,
    GTEBaseENV15,
    GTEBaseENV15Q,
    GTELargeENV15,
    GTELargeENV15Q,
    JINAV2SMALLEN,
    JINAV2BASEEN,
    JINAV3,
    SPLADEPPENV1,
    SPLADEPPENV2,
}
impl fmt::Display for ONNXModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[pyclass]
pub struct EmbeddingModel {
    pub inner: Arc<Embedder>,
}

#[pymethods]
impl EmbeddingModel {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, token=None, dtype=None))]
    fn from_pretrained_hf(
        model_id: Option<&str>,
        revision: Option<&str>,
        token: Option<&str>,
        dtype: Option<&Dtype>,
    ) -> PyResult<Self> {
        let model_id = model_id.unwrap_or("sentence-transformers/all-MiniLM-L12-v2");
        let dtype = match dtype {
            Some(Dtype::F16) => Some(embed_anything::Dtype::F16),
            Some(Dtype::F32) => Some(embed_anything::Dtype::F32),
            Some(Dtype::BF16) => Some(embed_anything::Dtype::BF16),
            Some(Dtype::INT8) => Some(embed_anything::Dtype::INT8),
            Some(Dtype::Q4) => Some(embed_anything::Dtype::Q4),
            Some(Dtype::UINT8) => Some(embed_anything::Dtype::UINT8),
            Some(Dtype::BNB4) => Some(embed_anything::Dtype::BNB4),
            Some(Dtype::Q4F16) => Some(embed_anything::Dtype::Q4F16),
            _ => None,
        };
        let model = Embedder::from_pretrained_hf(model_id, revision, token, dtype).unwrap();
        Ok(EmbeddingModel {
            inner: Arc::new(model),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (model, model_id,  api_key=None))]
    fn from_pretrained_cloud(
        model: &WhichModel,
        model_id: Option<&str>,
        api_key: Option<String>,
    ) -> PyResult<Self> {
        match model {
            WhichModel::OpenAI => {
                let model_id = model_id.unwrap_or("text-embedding-3-small");
                let model = Embedder::Text(TextEmbedder::OpenAI(
                    embed_anything::embeddings::cloud::openai::OpenAIEmbedder::new(
                        model_id.to_string(),
                        api_key,
                    ),
                ));
                Ok(EmbeddingModel {
                    inner: Arc::new(model),
                })
            }
            WhichModel::Cohere => {
                let model_id = model_id.unwrap_or("embed-english-v3.0");
                let model = Embedder::Text(TextEmbedder::Cohere(
                    embed_anything::embeddings::cloud::cohere::CohereEmbedder::new(
                        model_id.to_string(),
                        api_key,
                    ),
                ));
                Ok(EmbeddingModel {
                    inner: Arc::new(model),
                })
            }
            WhichModel::CohereVision => {
                let model_id = model_id.unwrap_or("embed-v4.0");
                let model = Embedder::Vision(Box::new(VisionEmbedder::Cohere(
                    embed_anything::embeddings::cloud::cohere::CohereEmbedder::new(
                        model_id.to_string(),
                        api_key,
                    ),
                )));
                Ok(EmbeddingModel {
                    inner: Arc::new(model),
                })
            }
            _ => panic!("Invalid model"),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (model, model_name=None, hf_model_id=None, revision=None, dtype=None, path_in_repo=None))]
    fn from_pretrained_onnx(
        model: &WhichModel,
        model_name: Option<&ONNXModel>,
        hf_model_id: Option<&str>,
        revision: Option<&str>,
        dtype: Option<&Dtype>,
        path_in_repo: Option<&str>,
    ) -> PyResult<Self> {
        let dtype = match dtype {
            Some(Dtype::Q4F16) => Some(embed_anything::Dtype::Q4F16),
            Some(Dtype::F16) => Some(embed_anything::Dtype::F16),
            Some(Dtype::INT8) => Some(embed_anything::Dtype::INT8),
            Some(Dtype::Q4) => Some(embed_anything::Dtype::Q4),
            Some(Dtype::UINT8) => Some(embed_anything::Dtype::UINT8),
            Some(Dtype::BNB4) => Some(embed_anything::Dtype::BNB4),
            Some(Dtype::F32) => Some(embed_anything::Dtype::F32),
            Some(Dtype::BF16) => Some(embed_anything::Dtype::BF16),
            None => None,
        };
        let model_name = model_name.map(|model_name| {
            embed_anything::embeddings::local::text_embedding::ONNXModel::from_str(
                &model_name.to_string(),
            )
            .unwrap()
        });
        match model {
            WhichModel::Bert => {
                let model = Embedder::Text(TextEmbedder::Bert(Box::new(
                    embed_anything::embeddings::local::ort_bert::OrtBertEmbedder::new(
                        model_name,
                        hf_model_id,
                        revision,
                        dtype,
                        path_in_repo,
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                )));
                Ok(EmbeddingModel {
                    inner: Arc::new(model),
                })
            }
            WhichModel::SparseBert => {
                let model = Embedder::Text(TextEmbedder::Bert(Box::new(
                    embed_anything::embeddings::local::ort_bert::OrtSparseBertEmbedder::new(
                        model_name,
                        hf_model_id,
                        revision,
                        path_in_repo,
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                )));
                Ok(EmbeddingModel {
                    inner: Arc::new(model),
                })
            }
            WhichModel::Jina => {
                let model = Embedder::Text(TextEmbedder::Jina(Box::new(
                    embed_anything::embeddings::local::ort_jina::OrtJinaEmbedder::new(
                        model_name,
                        hf_model_id,
                        revision,
                        dtype,
                        path_in_repo,
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                )));
                Ok(EmbeddingModel {
                    inner: Arc::new(model),
                })
            }
            WhichModel::ColBert => {
                let model = Embedder::Text(TextEmbedder::Bert(Box::new(
                    embed_anything::embeddings::local::colbert::OrtColbertEmbedder::new(
                        hf_model_id,
                        revision,
                        path_in_repo,
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                )));
                Ok(EmbeddingModel {
                    inner: Arc::new(model),
                })
            }
            _ => panic!("Invalid model"),
        }
    }

    #[pyo3(signature = (file_path, config=None, adapter=None))]
    pub fn embed_file(
        &self,
        file_path: &str,
        config: Option<&config::TextEmbedConfig>,
        adapter: Option<PyObject>,
    ) -> PyResult<Option<Vec<EmbedData>>> {
        embed_file(file_path, self, config, adapter)
    }

    #[pyo3(signature = (files, config=None, adapter=None))]
    pub fn embed_files_batch(
        &self,
        files: Vec<PathBuf>,
        config: Option<&config::TextEmbedConfig>,
        adapter: Option<PyObject>,
    ) -> PyResult<Option<Vec<EmbedData>>> {
        embed_files_batch(files, self, config, adapter)
    }

    #[pyo3(signature = (url, config=None, adapter=None))]
    pub fn embed_webpage(
        &self,
        url: &str,
        config: Option<&config::TextEmbedConfig>,
        adapter: Option<PyObject>,
    ) -> PyResult<Option<Vec<EmbedData>>> {
        embed_webpage(url.to_string(), self, config, adapter)
    }

    #[pyo3(signature = (directory, config=None, extensions=None, adapter=None))]
    pub fn embed_directory(
        &self,
        directory: PathBuf,
        config: Option<&config::TextEmbedConfig>,
        extensions: Option<Vec<String>>,
        adapter: Option<PyObject>,
    ) -> PyResult<Option<Vec<EmbedData>>> {
        embed_directory(directory, self, extensions, config, adapter)
    }

    #[pyo3(signature = (directory, config=None, adapter=None))]
    pub fn embed_image_directory(
        &self,
        directory: PathBuf,
        config: Option<&config::ImageEmbedConfig>,
        adapter: Option<PyObject>,
    ) -> PyResult<Option<Vec<EmbedData>>> {
        embed_image_directory(directory, self, config, adapter)
    }

    #[pyo3(signature = (query, config=None))]
    pub fn embed_query(
        &self,
        query: Vec<String>,
        config: Option<&config::TextEmbedConfig>,
    ) -> PyResult<Vec<EmbedData>> {
        embed_query(query, self, config)
    }

    #[pyo3(signature = (audio_file, audio_decoder, config=None))]
    pub fn embed_audio_file(
        &self,
        audio_file: String,
        audio_decoder: &mut AudioDecoderModel,
        config: Option<&config::TextEmbedConfig>,
    ) -> PyResult<Option<Vec<EmbedData>>> {
        embed_audio_file(audio_file, audio_decoder, self, config)
    }
}

#[pyclass]
pub struct AudioDecoderModel {
    pub inner: audio_processor::AudioDecoderModel,
}

#[pymethods]
impl AudioDecoderModel {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, model_type=None, quantized=None))]
    fn from_pretrained_hf(
        model_id: Option<&str>,
        revision: Option<&str>,
        model_type: Option<&str>,
        quantized: Option<bool>,
    ) -> PyResult<Self> {
        let model_id = model_id.unwrap_or("openai/whisper-tiny.en");
        let model_type = model_type.unwrap_or("tiny-en");
        let revision = revision.unwrap_or("main");
        let model = audio_processor::AudioDecoderModel::from_pretrained(
            Some(model_id),
            Some(revision),
            model_type,
            quantized.unwrap_or(false),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AudioDecoderModel { inner: model })
    }
}

#[pyfunction]
#[pyo3(signature = (query, embedder, config=None))]
pub fn embed_query(
    query: Vec<String>,
    embedder: &EmbeddingModel,
    config: Option<&config::TextEmbedConfig>,
) -> PyResult<Vec<EmbedData>> {
    let config = config.map(|c| &c.inner);
    let embedding_model = &embedder.inner;
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    Ok(rt.block_on(async {
        embed_anything::embed_query(
            &query.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
            embedding_model,
            Some(config.unwrap_or(&TextEmbedConfig::default())),
        )
        .await
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .unwrap()
        .into_iter()
        .map(|data| EmbedData { inner: data })
        .collect()
    }))
}

#[pyfunction]
#[pyo3(signature = (file_name, embedder, config=None, adapter=None))]
pub fn embed_file(
    file_name: &str,
    embedder: &EmbeddingModel,
    config: Option<&config::TextEmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let config = config.map(|c| &c.inner);
    let embedding_model = &embedder.inner;
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    if !Path::new(file_name).exists() {
        // check if the file exists other wise return a "File not found" error with PyValueError
        return Err(PyFileNotFoundError::new_err(format!(
            "File not found: {:?}",
            file_name
        )));
    };
    let adapter = match adapter {
        Some(adapter) => {
            let callback = move |data: Vec<embed_anything::embeddings::embed::EmbedData>| {
                Python::with_gil(|py| {
                    let upsert_fn = adapter.getattr(py, "upsert").unwrap();
                    let converted_data = data
                        .into_iter()
                        .map(|data| EmbedData { inner: data })
                        .collect::<Vec<EmbedData>>();
                    upsert_fn
                        .call1(py, (converted_data,))
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                        .unwrap();
                });
            };
            Some(callback)
        }
        None => None,
    };

    let embeddings = rt
        .block_on(async {
            embed_anything::embed_file(
                file_name,
                embedding_model,
                config,
                adapter.map(|f| {
                    Box::new(f)
                        as Box<
                            dyn FnOnce(Vec<embed_anything::embeddings::embed::EmbedData>)
                                + Send
                                + Sync,
                        >
                }),
            )
            .await
        })
        .map_err(|e| match e.downcast_ref::<FileLoadingError>() {
            Some(FileLoadingError::FileNotFound(file)) => {
                PyFileNotFoundError::new_err(file.clone())
            }
            Some(FileLoadingError::UnsupportedFileType(file)) => {
                PyValueError::new_err(file.clone())
            }
            None => PyValueError::new_err(e.to_string()),
        })?;

    Ok(embeddings.map(|embs| {
        embs.into_iter()
            .map(|data| EmbedData { inner: data })
            .collect()
    }))
}

#[pyfunction]
#[pyo3(signature = (files, embedder, config=None, adapter=None))]
pub fn embed_files_batch(
    files: Vec<PathBuf>,
    embedder: &EmbeddingModel,
    config: Option<&config::TextEmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let config = config.map(|c| &c.inner);
    let embedding_model = &embedder.inner;
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let adapter = match adapter {
        Some(adapter) => {
            let callback = move |data: Vec<embed_anything::embeddings::embed::EmbedData>| {
                Python::with_gil(|py| {
                    let upsert_fn = adapter.getattr(py, "upsert").unwrap();
                    let converted_data = data
                        .into_iter()
                        .map(|data| EmbedData { inner: data })
                        .collect::<Vec<EmbedData>>();
                    upsert_fn
                        .call1(py, (converted_data,))
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                        .unwrap();
                });
            };
            Some(callback)
        }
        None => None,
    };

    let embeddings = rt
        .block_on(async {
            embed_anything::embed_files_batch(
                files,
                embedding_model,
                config,
                adapter.map(|f| {
                    Box::new(f)
                        as Box<
                            dyn FnMut(Vec<embed_anything::embeddings::embed::EmbedData>)
                                + Send
                                + Sync,
                        >
                }),
            )
            .await
        })
        .map_err(|e| match e.downcast_ref::<FileLoadingError>() {
            Some(FileLoadingError::FileNotFound(file)) => {
                PyFileNotFoundError::new_err(file.clone())
            }
            Some(FileLoadingError::UnsupportedFileType(file)) => {
                PyValueError::new_err(file.clone())
            }
            None => PyValueError::new_err(e.to_string()),
        })?;

    Ok(embeddings.map(|embs| {
        embs.into_iter()
            .map(|data| EmbedData { inner: data })
            .collect()
    }))
}

#[pyfunction]
#[pyo3(signature = (audio_file, audio_decoder, embedder, text_embed_config=None))]
pub fn embed_audio_file(
    audio_file: String,
    audio_decoder: &mut AudioDecoderModel,
    embedder: &EmbeddingModel,
    text_embed_config: Option<&config::TextEmbedConfig>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let config = text_embed_config.map(|c| &c.inner);
    let embedding_model = &embedder.inner;
    let audio_decoder = &mut audio_decoder.inner;
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let data = rt.block_on(async {
        emb_audio(audio_file, audio_decoder, embedding_model, config)
            .await
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .unwrap()
            .map(|data| {
                data.into_iter()
                    .map(|data| EmbedData { inner: data })
                    .collect::<Vec<_>>()
            })
    });
    Ok(data)
}

#[pyfunction]
#[pyo3(signature = (directory, embedder, extensions=None, config=None, adapter = None))]
pub fn embed_directory(
    directory: PathBuf,
    embedder: &EmbeddingModel,
    extensions: Option<Vec<String>>,
    config: Option<&config::TextEmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let config = config.map(|c| &c.inner);
    let embedding_model = &embedder.inner;

    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let adapter = match adapter {
        Some(adapter) => {
            let callback = move |data: Vec<embed_anything::embeddings::embed::EmbedData>| {
                Python::with_gil(|py| {
                    let upsert_fn = adapter.getattr(py, "upsert").unwrap();
                    let converted_data = data
                        .into_iter()
                        .map(|data| EmbedData { inner: data })
                        .collect::<Vec<EmbedData>>();
                    upsert_fn
                        .call1(py, (converted_data,))
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                        .unwrap();
                });
            };
            Some(callback)
        }
        None => None,
    };

    let data = rt.block_on(async {
        embed_anything::embed_directory_stream(
            directory,
            embedding_model,
            extensions,
            config,
            adapter.map(|f| {
                Box::new(f)
                    as Box<
                        dyn FnMut(Vec<embed_anything::embeddings::embed::EmbedData>) + Send + Sync,
                    >
            }),
        )
        .await
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .unwrap()
        .map(|data| {
            data.into_iter()
                .map(|data| EmbedData { inner: data })
                .collect::<Vec<_>>()
        })
    });
    Ok(data)
}

#[pyfunction]
#[pyo3(signature = (directory, embedder, config=None, adapter = None))]
pub fn embed_image_directory(
    directory: PathBuf,
    embedder: &EmbeddingModel,
    config: Option<&config::ImageEmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let embedding_model = &embedder.inner;
    let config = config.map(|c| &c.inner);
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let adapter = match adapter {
        Some(adapter) => {
            let callback = move |data: Vec<embed_anything::embeddings::embed::EmbedData>| {
                Python::with_gil(|py| {
                    let upsert_fn = adapter.getattr(py, "upsert").unwrap();
                    let converted_data = data
                        .into_iter()
                        .map(|data| EmbedData { inner: data })
                        .collect::<Vec<EmbedData>>();
                    upsert_fn
                        .call1(py, (converted_data,))
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                        .unwrap();
                });
            };
            Some(callback)
        }
        None => None,
    };

    let data = rt.block_on(async {
        embed_anything::embed_image_directory(
            directory,
            embedding_model,
            config,
            adapter.map(|f| {
                Box::new(f)
                    as Box<
                        dyn FnMut(Vec<embed_anything::embeddings::embed::EmbedData>) + Send + Sync,
                    >
            }),
        )
        .await
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .unwrap()
        .map(|data| {
            data.into_iter()
                .map(|data| EmbedData { inner: data })
                .collect::<Vec<_>>()
        })
    });
    Ok(data)
}

#[pyfunction]
#[pyo3(signature = (url, embedder, config=None, adapter = None))]
pub fn embed_webpage(
    url: String,
    embedder: &EmbeddingModel,
    config: Option<&config::TextEmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let embedding_model = &embedder.inner;
    let config = config.map(|c| &c.inner);
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let adapter = match adapter {
        Some(adapter) => {
            let callback = move |data: Vec<embed_anything::embeddings::embed::EmbedData>| {
                Python::with_gil(|py| {
                    let upsert_fn = adapter.getattr(py, "upsert").unwrap();
                    let converted_data = data
                        .into_iter()
                        .map(|data| EmbedData { inner: data })
                        .collect::<Vec<EmbedData>>();
                    upsert_fn
                        .call1(py, (converted_data,))
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                        .unwrap();
                });
            };
            Some(callback)
        }
        None => None,
    };

    let data = rt.block_on(async {
        embed_anything::embed_webpage(
            url,
            embedding_model,
            config,
            adapter.map(|f| {
                Box::new(f)
                    as Box<
                        dyn FnOnce(Vec<embed_anything::embeddings::embed::EmbedData>) + Send + Sync,
                    >
            }),
        )
        .await
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .unwrap()
        .map(|data| {
            data.into_iter()
                .map(|data| EmbedData { inner: data })
                .collect::<Vec<_>>()
        })
    });
    Ok(data)
}

#[pymodule]
fn _embed_anything(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_file, m)?)?;
    m.add_function(wrap_pyfunction!(embed_directory, m)?)?;
    m.add_function(wrap_pyfunction!(embed_image_directory, m)?)?;
    m.add_function(wrap_pyfunction!(embed_query, m)?)?;
    m.add_function(wrap_pyfunction!(embed_webpage, m)?)?;
    m.add_function(wrap_pyfunction!(embed_audio_file, m)?)?;
    m.add_class::<ColpaliModel>()?;
    m.add_class::<ColbertModel>()?;
    m.add_class::<EmbeddingModel>()?;
    m.add_class::<AudioDecoderModel>()?;
    m.add_class::<WhichModel>()?;
    m.add_class::<EmbedData>()?;
    m.add_class::<config::TextEmbedConfig>()?;
    m.add_class::<ONNXModel>()?;
    m.add_class::<Reranker>()?;
    m.add_class::<Dtype>()?;
    m.add_class::<RerankerResult>()?;
    m.add_class::<DocumentRank>()?;
    m.add_class::<s3_client::S3Client>()?;
    m.add_class::<s3_client::S3File>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_file() {
        let embedder = EmbeddingModel::from_pretrained_hf(None, None, None, None).unwrap();
        let embeddings = embedder
            .embed_query(vec!["Hello, world!".to_string()], None)
            .unwrap();
        assert!(!embeddings.is_empty());
    }
}
