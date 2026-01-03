use embed_anything::s3_loader::{S3Client as RustS3Client, S3File as RustS3File};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::runtime::Builder;

/// Represents a file fetched from S3
#[pyclass]
pub struct S3File {
    pub inner: RustS3File,
}

#[pymethods]
impl S3File {
    /// Save the file to the local filesystem
    ///
    /// Args:
    ///     file_path: Optional local file path. If not provided, uses the S3 key basename in current directory
    ///
    /// Returns:
    ///     str: The file path where the file was saved
    #[pyo3(signature = (file_path=None))]
    pub fn save_file(&self, file_path: Option<String>) -> PyResult<String> {
        self.inner
            .save_file(file_path.as_deref())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get the S3 key
    #[getter]
    pub fn key(&self) -> String {
        self.inner.key().to_string()
    }

    /// Get the file contents as bytes
    #[getter]
    pub fn bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.as_bytes())
    }

    /// Get the size of the file in bytes
    #[getter]
    pub fn size(&self) -> usize {
        self.inner.as_bytes().len()
    }

    fn __repr__(&self) -> String {
        format!("S3File({} bytes)", self.size())
    }

    fn __len__(&self) -> usize {
        self.size()
    }
}

/// S3Client for fetching files from AWS S3 buckets
#[pyclass]
pub struct S3Client {
    pub inner: RustS3Client,
}

#[pymethods]
impl S3Client {
    /// Create a new S3Client with AWS credentials
    ///
    /// Args:
    ///     access_key_id: AWS access key ID
    ///     secret_access_key: AWS secret access key
    ///     region: AWS region (e.g., "us-east-1")
    #[new]
    pub fn new(
        access_key_id: String,
        secret_access_key: String,
        region: String,
    ) -> Self {
        S3Client {
            inner: RustS3Client::new(
                access_key_id,
                secret_access_key,
                region,
            ),
        }
    }

    /// Create an S3Client using environment variables
    ///
    /// Reads credentials from:
    /// - AWS_ACCESS_KEY_ID
    /// - AWS_SECRET_ACCESS_KEY
    /// - AWS_REGION (optional, defaults to "us-east-1")
    #[staticmethod]
    pub fn from_env() -> PyResult<Self> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| PyValueError::new_err("AWS_ACCESS_KEY_ID environment variable not set"))?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| {
            PyValueError::new_err("AWS_SECRET_ACCESS_KEY environment variable not set")
        })?;
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

        Ok(S3Client {
            inner: RustS3Client::new(access_key, secret_key, region),
        })
    }

    /// Fetch a file from S3 and return it as an S3File
    ///
    /// Args:
    ///     bucket_name: The S3 bucket name
    ///     key: The S3 object key (file path)
    ///
    /// Returns:
    ///     S3File: The file object with methods to save or access bytes
    pub fn get_file_from_s3(&self, bucket_name: String, key: String) -> PyResult<S3File> {
        let rt = Builder::new_multi_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let file = self
                .inner
                .get_file_from_s3(&bucket_name, &key)
                .await
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(S3File { inner: file })
        })
    }

    fn __repr__(&self) -> String {
        format!("S3Client(region='{}')", self.inner.region)
    }
}
