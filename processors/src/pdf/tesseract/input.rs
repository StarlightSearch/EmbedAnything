use crate::pdf::tesseract::error::{TessError, TessResult};
use image::DynamicImage;
use std::{
    collections::HashMap,
    fmt::{self},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Args {
    pub lang: String,
    pub config_variables: HashMap<String, String>,
    pub dpi: Option<i32>,
    pub psm: Option<i32>,
    pub oem: Option<i32>,
    pub path: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            lang: "eng".into(),
            config_variables: HashMap::new(),
            dpi: Some(150),
            psm: Some(3),
            oem: Some(3),
            path: None,
        }
    }
}

impl Args {
    pub fn with_path(mut self, path: Option<&str>) -> Self {
        self.path = path.map(|p| p.to_string());
        self
    }

    pub(crate) fn get_config_variable_args(&self) -> Vec<String> {
        self.config_variables
            .iter()
            .map(|(key, value)| format!("{}={}", key, value))
            .collect::<Vec<_>>()
    }
}

#[derive(Debug)]
pub struct Image {
    data: InputData,
}

impl Image {
    pub fn from_path<P: Into<PathBuf>>(path: P) -> TessResult<Self> {
        let path = path.into();
        Self::check_image_format(&path)?;
        Ok(Self {
            data: InputData::Path(path),
        })
    }

    fn check_image_format(path: &Path) -> TessResult<()> {
        let binding = path
            .extension()
            .ok_or(TessError::ImageFormatError)?
            .to_str()
            .ok_or(TessError::ImageFormatError)?
            .to_uppercase();
        if matches!(
            binding.as_str(),
            "JPEG" | "JPG" | "PNG" | "PBM" | "PGM" | "PPM" | "TIFF" | "BMP" | "GIF" | "WEBP"
        ) {
            Ok(())
        } else {
            Err(TessError::ImageFormatError)
        }
    }

    pub fn from_dynamic_image(image: &DynamicImage) -> TessResult<Self> {
        //Store Image as Tempfile
        let tempfile = tempfile::Builder::new()
            .prefix("rusty-tesseract")
            .suffix(".png")
            .tempfile()
            .map_err(|e| TessError::TempfileError(e.to_string()))?;
        let path = tempfile.path();
        image
            .save_with_format(path, image::ImageFormat::Png)
            .map_err(|e| TessError::DynamicImageError(e.to_string()))?;

        Ok(Self {
            data: InputData::Image(tempfile),
        })
    }

    pub fn get_image_path(&self) -> TessResult<&str> {
        match &self.data {
            InputData::Path(x) => x.to_str(),
            InputData::Image(x) => x.path().to_str(),
        }
        .ok_or(TessError::ImageNotFoundError)
    }
}

#[derive(Debug)]
enum InputData {
    Path(PathBuf),
    Image(tempfile::NamedTempFile),
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get_image_path().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::Image;
    use image::ImageReader;

    #[test]
    fn test_from_path() {
        let input = Image::from_path("../test_files/clip/cat1.jpg").unwrap();

        assert_eq!(
            input.get_image_path().unwrap(),
            "../test_files/clip/cat1.jpg"
        )
    }

    #[test]
    fn test_from_dynamic_image() {
        let img = ImageReader::open("../test_files/clip/cat1.jpg")
            .unwrap()
            .decode()
            .unwrap();

        let input = Image::from_dynamic_image(&img).unwrap();

        let temppath = input.get_image_path().unwrap();

        let tempimg = ImageReader::open(temppath).unwrap().decode().unwrap();

        assert_eq!(img, tempimg);
    }
}
