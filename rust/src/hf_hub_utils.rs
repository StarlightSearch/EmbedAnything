use std::path::PathBuf;

use anyhow::{Error, Result};
use hf_hub::{split_id, HFClient, HFClientSync, HFRepositorySync, RepoTypeModel};

pub type ModelRepo = HFRepositorySync<RepoTypeModel>;

/// Build a blocking Hugging Face Hub client, optionally overriding the token.
pub fn build_client(token: Option<&str>) -> Result<HFClientSync> {
    let client = match token {
        Some(token) => HFClient::builder().token(token).build().map_err(Error::msg)?,
        None => HFClient::new().map_err(Error::msg)?,
    };
    HFClientSync::from_inner(client).map_err(Error::msg)
}

/// Return a model repository handle for `model_id` (`owner/name` or short id).
pub fn model_repo(client: &HFClientSync, model_id: &str) -> ModelRepo {
    let (owner, name) = split_id(model_id);
    client.model(owner, name)
}

/// Download a single file from a model repo into the local cache.
pub fn download_file(
    repo: &ModelRepo,
    filename: &str,
    revision: Option<&str>,
) -> Result<PathBuf> {
    repo.download_file()
        .filename(filename)
        .maybe_revision(revision.map(|s| s.to_string()))
        .send()
        .map_err(Error::msg)
}
