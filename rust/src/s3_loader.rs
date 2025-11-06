//! S3 file loading utilities.
//!
//! Provides functionality for fetching files from AWS S3 buckets
//! for embedding processing.

use anyhow::{Context, Result};
use aws_config::BehaviorVersion;
use aws_credential_types::Credentials;
use aws_sdk_s3::Client;

/// Represents a file fetched from S3
#[derive(Debug, Clone)]
pub struct S3File {
    pub bytes: Vec<u8>,
    pub key: String,
}

impl S3File {
    /// Create a new S3File from bytes and S3 key
    pub fn new(bytes: Vec<u8>, key: String) -> Self {
        Self { bytes, key }
    }

    /// Save the file to the local filesystem
    ///
    /// # Arguments
    ///
    /// * `file_path` - Optional local file path. If None, uses the S3 key basename in current directory
    ///
    /// # Returns
    ///
    /// A `Result` containing the file path on success
    pub fn save_file(&self, file_path: Option<&str>) -> Result<String> {
        let path = match file_path {
            Some(p) => p.to_string(),
            None => {
                // Extract filename from S3 key (last part after /)
                let filename = self
                    .key
                    .split('/')
                    .last()
                    .unwrap_or(&self.key)
                    .to_string();
                filename
            }
        };

        std::fs::write(&path, &self.bytes)
            .context(format!("Failed to write file to: {}", path))?;
        Ok(path)
    }

    /// Get the bytes as a slice
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Convert to Vec<u8>
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Get the S3 key
    pub fn key(&self) -> &str {
        &self.key
    }
}

/// S3 client for fetching files from AWS S3 buckets
#[derive(Debug, Clone)]
pub struct S3Client {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub region: String,
}

impl S3Client {
    /// Create a new S3Client with the specified credentials and region
    ///
    /// # Arguments
    ///
    /// * `access_key_id` - AWS access key ID
    /// * `secret_access_key` - AWS secret access key
    /// * `region` - AWS region (e.g., "us-east-1")
    pub fn new(access_key_id: String, secret_access_key: String, region: String) -> Self {
        Self {
            access_key_id,
            secret_access_key,
            region,
        }
    }

    /// Fetch a file from S3 and return it as an S3File
    ///
    /// # Arguments
    ///
    /// * `bucket_name` - The S3 bucket name
    /// * `key` - The S3 object key (file path)
    ///
    /// # Returns
    ///
    /// A `Result` containing an `S3File` on success
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use embed_anything::s3_loader::S3Client;
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = S3Client::new("key".to_string(), "secret".to_string(), "us-east-1".to_string());
    /// let file = client.get_file_from_s3("my-bucket", "file.txt").await?;
    /// file.save_file("/tmp/downloaded.txt")?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_file_from_s3(&self, bucket_name: &str, key: &str) -> Result<S3File> {
        let credentials = Credentials::new(
            &self.access_key_id,
            &self.secret_access_key,
            None,
            None,
            "static",
        );

        let config = aws_config::defaults(BehaviorVersion::latest())
            .credentials_provider(credentials)
            .region(aws_config::Region::new(self.region.clone()))
            .load()
            .await;

        let client = Client::new(&config);

        let response = client
            .get_object()
            .bucket(bucket_name)
            .key(key)
            .send()
            .await
            .context(format!(
                "Failed to fetch object from S3: bucket={}, key={}",
                bucket_name, key
            ))?;

        let body = response.body.collect().await?;
        Ok(S3File::new(body.into_bytes().to_vec(), key.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires valid AWS credentials
    async fn test_get_file_from_s3() {
        // To run this test:
        // 1. Set valid AWS credentials
        // 2. Create a test bucket and upload a test file
        // 3. Run: cargo test --lib s3_loader -- --ignored

        let client = S3Client::new(
            "YOUR_ACCESS_KEY".to_string(),
            "YOUR_SECRET_KEY".to_string(),
            "us-east-1".to_string(),
        );
        let file = client
            .get_file_from_s3("your-bucket", "test.txt")
            .await
            .unwrap();
        println!("{}", String::from_utf8_lossy(file.as_bytes()));
    }

    #[tokio::test]
    #[ignore] // Requires valid AWS credentials
    async fn test_download_and_save() {
        use tempdir::TempDir;

        let client = S3Client::new(
            "YOUR_ACCESS_KEY".to_string(),
            "YOUR_SECRET_KEY".to_string(),
            "us-east-1".to_string(),
        );

        let file = client
            .get_file_from_s3("your-bucket", "test.txt")
            .await
            .unwrap();

        let temp_dir = TempDir::new("test").unwrap();
        let path = temp_dir.path().join("downloaded.txt");
        let saved_path = file.save_file(Some(path.to_str().unwrap())).unwrap();

        assert_eq!(saved_path, path.to_str().unwrap());
        assert!(std::path::Path::new(&saved_path).exists());
    }

    #[test]
    fn test_s3file_creation() {
        let bytes = vec![1, 2, 3, 4, 5];
        let file = S3File::new(bytes.clone(), "test/file.txt".to_string());
        assert_eq!(file.as_bytes(), &bytes);
        assert_eq!(file.key(), "test/file.txt");
        assert_eq!(file.clone().into_bytes(), bytes);
    }

    #[test]
    fn test_s3file_save_with_default_name() {
        use tempdir::TempDir;

        let temp_dir = TempDir::new("test").unwrap();
        std::env::set_current_dir(temp_dir.path()).unwrap();

        let bytes = b"test content".to_vec();
        let file = S3File::new(bytes.clone(), "path/to/myfile.txt".to_string());

        // Save without specifying path - should use basename
        let saved_path = file.save_file(None).unwrap();
        assert_eq!(saved_path, "myfile.txt");
        assert!(std::path::Path::new(&saved_path).exists());

        let content = std::fs::read(&saved_path).unwrap();
        assert_eq!(content, bytes);
    }

    #[test]
    fn test_s3file_save_with_custom_path() {
        use tempdir::TempDir;

        let bytes = b"test content".to_vec();
        let file = S3File::new(bytes.clone(), "original.txt".to_string());

        let temp_dir = TempDir::new("test").unwrap();
        let path = temp_dir.path().join("custom.txt");

        let saved_path = file.save_file(Some(path.to_str().unwrap())).unwrap();
        assert_eq!(saved_path, path.to_str().unwrap());

        let content = std::fs::read(&saved_path).unwrap();
        assert_eq!(content, bytes);
    }

    #[test]
    fn test_s3_client_creation() {
        let client = S3Client::new(
            "test_key".to_string(),
            "test_secret".to_string(),
            "us-west-2".to_string(),
        );
        assert_eq!(client.access_key_id, "test_key");
        assert_eq!(client.secret_access_key, "test_secret");
        assert_eq!(client.region, "us-west-2");
    }

}
