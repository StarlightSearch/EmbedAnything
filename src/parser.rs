use std::{collections::HashSet, io::Error, path::PathBuf};


use regex::Regex;
use walkdir::WalkDir;

pub struct FileParser {
    pub files: Vec<String>,
}

impl Default for FileParser {
    fn default() -> Self {
        Self::new()
    }
}

impl FileParser {
    pub fn new() -> Self {
        Self { files: Vec::new() }
    }

    pub fn get_pdf_files(&mut self, directory_path: &PathBuf) -> Result<Vec<String>, Error> {
        let pdf_extension_regex = Regex::new(r#"\.pdf$"#).unwrap();

        let pdf_files: Vec<String> = WalkDir::new(directory_path)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| pdf_extension_regex.is_match(entry.file_name().to_str().unwrap_or("")))
            .map(|entry| {
                let absolute_path = entry
                    .path()
                    .canonicalize()
                    .unwrap_or_else(|_| entry.path().to_path_buf());
                absolute_path.to_string_lossy().to_string()
            })
            .collect();
        self.files = pdf_files;

        Ok(self.files.clone())
    }

    pub fn get_image_paths(&mut self, directory_path: &PathBuf) -> Result<Vec<String>, Error> {
        let image_regex = Regex::new(r".*\.(png|jpg|jpeg|gif|bmp|tiff|webp)$").unwrap();
    
        let image_paths: Vec<String> = WalkDir::new(directory_path)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| image_regex.is_match(entry.file_name().to_str().unwrap_or("")))
            .map(|entry| {
                let absolute_path = entry
                    .path()
                    .canonicalize()
                    .unwrap_or_else(|_| entry.path().to_path_buf());
                absolute_path.to_string_lossy().to_string()
            })
            .collect();
    
        self.files = image_paths;
        Ok(self.files.clone())
    }

    pub fn get_files_to_index(&self, indexed_files: &HashSet<String>) -> Vec<String> {
        let files = self
            .files
            .iter()
            .filter(|file| !indexed_files.contains(*file))
            .map(|f| f.to_string())
            .collect::<Vec<_>>();
        files
    }


}
