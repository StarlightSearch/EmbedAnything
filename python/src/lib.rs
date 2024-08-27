pub mod config;

use embed_anything;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::{collections::HashMap, path::PathBuf};

#[pyclass]
pub struct EmbedData {
    pub inner: embed_anything::embeddings::embed::EmbedData,
}

#[pymethods]
impl EmbedData {
    #[getter(embedding)]
    fn embedding(&self) -> Vec<f32> {
        self.inner.embedding.clone()
    }

    #[getter(text)]
    fn text(&self) -> Option<String> {
        self.inner.text.clone()
    }

    #[getter(metadata)]
    fn metadata(&self) -> Option<HashMap<String, String>> {
        self.inner.metadata.clone()
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
        format!("<class 'EmbedData'>")
    }
}

#[pyfunction]
#[pyo3(signature = (query, embeder, config=None))]
pub fn embed_query(
    query: Vec<String>,
    embeder: &str,
    config: Option<&config::EmbedConfig>,
) -> PyResult<Vec<EmbedData>> {
    let config = config.map(|c| &c.inner);
    Ok(embed_anything::embed_query(query, embeder, config)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_iter()
        .map(|data| EmbedData { inner: data })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (file_name, embeder, config=None, adapter=None))]
pub fn embed_file(
    file_name: &str,
    embeder: &str,
    config: Option<&config::EmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let config = config.map(|c| &c.inner);

    match adapter {
        Some(adapter) => Python::with_gil(|py| {
            let callback = |data: Vec<embed_anything::embeddings::embed::EmbedData>| {
                let upsert_fn = adapter.getattr(py, "upsert").unwrap();
                let converted_data = data
                    .into_iter()
                    .map(|data| EmbedData { inner: data })
                    .collect::<Vec<EmbedData>>();
                upsert_fn
                    .call1(py, (converted_data,))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
                    .unwrap();
            };

            Ok(
                embed_anything::embed_file(file_name, embeder, config, Some(callback))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .map(|data| {
                        data.into_iter()
                            .map(|data| EmbedData { inner: data })
                            .collect()
                    }),
            )
        }),
        None => Ok(embed_anything::embed_file(
            file_name,
            embeder,
            config,
            None::<fn(Vec<embed_anything::embeddings::embed::EmbedData>)>,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .map(|data| {
            data.into_iter()
                .map(|data| EmbedData { inner: data })
                .collect()
        })),
    }
}

#[pyfunction]
#[pyo3(signature = (directory, embeder, extensions=None, config=None, adapter = None))]
pub fn embed_directory(
    directory: PathBuf,
    embeder: &str,
    extensions: Option<Vec<String>>,
    config: Option<&config::EmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let config = config.map(|c| &c.inner);

    match adapter {
        Some(adapter) => Python::with_gil(|py| {
            let callback = |data: Vec<embed_anything::embeddings::embed::EmbedData>| {
                let upsert_fn = adapter.getattr(py, "upsert").unwrap();
                let converted_data = data
                    .into_iter()
                    .map(|data| EmbedData { inner: data })
                    .collect::<Vec<EmbedData>>();
                upsert_fn
                    .call1(py, (converted_data,))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
                    .unwrap();
            };

            Ok(embed_anything::embed_directory(
                directory,
                embeder,
                extensions,
                config,
                Some(callback),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .map(|data| {
                data.into_iter()
                    .map(|data| EmbedData { inner: data })
                    .collect()
            }))
        }),
        None => Ok(embed_anything::embed_directory(
            directory,
            embeder,
            extensions,
            config,
            None::<fn(Vec<embed_anything::embeddings::embed::EmbedData>)>,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .map(|data| {
            data.into_iter()
                .map(|data| EmbedData { inner: data })
                .collect()
        })),
    }
}

#[pyfunction]
#[pyo3(signature = (url, embeder, config=None, adapter = None))]
pub fn embed_webpage(
    url: String,
    embeder: &str,
    config: Option<&config::EmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let config = config.map(|c| &c.inner);

    match adapter {
        Some(adapter) => Python::with_gil(|py| {
            let callback = |data: Vec<embed_anything::embeddings::embed::EmbedData>| {
                let upsert_fn = adapter.getattr(py, "upsert").unwrap();
                let converted_data = data
                    .into_iter()
                    .map(|data| EmbedData { inner: data })
                    .collect::<Vec<EmbedData>>();
                upsert_fn
                    .call1(py, (converted_data,))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
                    .unwrap();
            };

            Ok(embed_anything::embed_webpage(
                url,
                embeder,
                config,
                Some(callback),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .map(|data| {
                data.into_iter()
                    .map(|data| EmbedData { inner: data })
                    .collect()
            }))
        }),
        None => Ok(embed_anything::embed_webpage(
            url,
            embeder,
            config,
            None::<fn(Vec<embed_anything::embeddings::embed::EmbedData>)>,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .map(|data| {
            data.into_iter()
                .map(|data| EmbedData { inner: data })
                .collect()
        })),
    }}

#[pymodule]
fn _embed_anything(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_file, m)?)?;
    m.add_function(wrap_pyfunction!(embed_directory, m)?)?;
    m.add_function(wrap_pyfunction!(embed_query, m)?)?;
    m.add_function(wrap_pyfunction!(embed_webpage, m)?)?;
    m.add_class::<EmbedData>()?;
    m.add_class::<config::BertConfig>()?;
    m.add_class::<config::ClipConfig>()?;
    m.add_class::<config::CloudConfig>()?;
    m.add_class::<config::EmbedConfig>()?;
    m.add_class::<config::AudioDecoderConfig>()?;
    m.add_class::<config::JinaConfig>()?;

    Ok(())
}
