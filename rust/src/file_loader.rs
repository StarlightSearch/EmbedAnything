use std::{collections::HashSet, io::Error, path::PathBuf};

use regex::Regex;
use walkdir::WalkDir;
// use tokio::fs;

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

    pub fn get_text_files(
        &mut self,
        directory_path: &PathBuf,
        extensions: Option<Vec<String>>,
    ) -> Result<Vec<String>, Error> {
        let extension_regex = match extensions {
            Some(exts) => Regex::new(&format!(r"\.({})$", exts.join("|"))).unwrap(),
            None => Regex::new(r"\.(pdf|md|txt)$").unwrap(),
        };

        let entries = std::fs::read_dir(directory_path)?;
        let mut files = Vec::new();

        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let file_name = entry.file_name();
                if extension_regex.is_match(file_name.to_str().unwrap_or("")) {
                    let absolute_path =
                        std::fs::canonicalize(entry.path()).unwrap_or_else(|_| entry.path());
                    files.push(absolute_path.to_string_lossy().to_string());
                }
            }
        }

        self.files = files;
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

    pub fn get_audio_files(&mut self, directory_path: &PathBuf) -> Result<Vec<String>, Error> {
        let audio_regex = Regex::new(r".*\.(wav)$").unwrap();

        let audio_paths: Vec<String> = WalkDir::new(directory_path)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| audio_regex.is_match(entry.file_name().to_str().unwrap_or("")))
            .map(|entry| {
                let absolute_path = entry
                    .path()
                    .canonicalize()
                    .unwrap_or_else(|_| entry.path().to_path_buf());
                absolute_path.to_string_lossy().to_string()
            })
            .collect();

        self.files = audio_paths;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempdir::TempDir;

    #[test]
    fn test_get_text_files() {
        let temp_dir = TempDir::new("example").unwrap();
        let pdf_file = temp_dir.path().join("test.pdf");
        let txt_file = temp_dir.path().join("test.txt");
        let markdown_file = (0..2)
            .map(|f| temp_dir.path().join(format!("test{}.md", f)))
            .collect::<Vec<_>>();
        let _image_file = temp_dir.path().join("image.jpg");

        File::create(&pdf_file).unwrap();
        File::create(&txt_file).unwrap();
        markdown_file.iter().for_each(|f| {
            File::create(f).unwrap();
        });

        let mut file_parser = FileParser::new();
        let pdf_files = file_parser
            .get_text_files(
                &PathBuf::from(temp_dir.path()),
                Some(vec!["pdf".to_string()]),
            )
            .unwrap();
        let text_files = file_parser
            .get_text_files(
                &PathBuf::from(temp_dir.path()),
                Some(vec!["txt".to_string()]),
            )
            .unwrap();
        let markdown_files = file_parser
            .get_text_files(
                &PathBuf::from(temp_dir.path()),
                Some(vec!["md".to_string()]),
            )
            .unwrap();

        assert_eq!(pdf_files.len(), 1);
        assert_eq!(text_files.len(), 1);
        assert_eq!(markdown_files.len(), 2);
        assert_eq!(
            pdf_files[0],
            pdf_file
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .to_string()
        );
        assert_eq!(
            text_files[0],
            txt_file
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .to_string()
        );
        assert_eq!(
            markdown_files.clone().sort(),
            markdown_file
                .iter()
                .map(|f| f.canonicalize().unwrap().to_string_lossy().to_string())
                .collect::<Vec<String>>()
                .clone()
                .sort()
        );
    }

    #[test]
    fn test_get_image_paths() {
        let temp_dir = TempDir::new("example").unwrap();
        let image_file = temp_dir.path().join("image.jpg");
        let _pdf_file = temp_dir.path().join("test.pdf");
        File::create(&image_file).unwrap();

        let mut file_parser = FileParser::new();
        let image_files = file_parser
            .get_image_paths(&PathBuf::from(temp_dir.path()))
            .unwrap();
        assert_eq!(image_files.len(), 1);
        assert_eq!(
            image_files[0],
            image_file
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .to_string()
        );
    }

    #[test]
    fn test_get_audio_paths() {
        let mut file_parser = FileParser::new();
        let audio_files = file_parser
            .get_audio_files(&PathBuf::from("test_files"))
            .unwrap();

        assert_eq!(audio_files.len(), 2);
    }

    #[test]
    fn test_get_files_to_index() {
        let temp_dir = TempDir::new("example").unwrap();
        let pdf_file = temp_dir.path().join("test.pdf");
        let image_file = temp_dir.path().join("test.jpg");
        File::create(&pdf_file).unwrap();
        File::create(&image_file).unwrap();

        let mut file_parser = FileParser::new();
        file_parser
            .get_text_files(&PathBuf::from(temp_dir.path()), None)
            .unwrap();
        file_parser
            .get_image_paths(&PathBuf::from(temp_dir.path()))
            .unwrap();

        let indexed_files = vec![pdf_file.to_string_lossy().to_string()];
        let files_to_index = file_parser.get_files_to_index(&indexed_files.into_iter().collect());
        assert_eq!(files_to_index.len(), 1);
        assert_eq!(
            files_to_index[0],
            image_file
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .to_string()
        );
    }
}
