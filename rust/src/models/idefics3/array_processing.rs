use std::collections::HashMap;

use anyhow::{Error, Ok};
use hf_hub::{api::sync::Api, Repo, RepoType};
use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};
use ndarray::{s, Array2};
use regex::Regex;
use serde::Deserialize;
use tokenizers::{AddedToken, PaddingParams, Tokenizer, TruncationParams};

const MAX_IMAGE_SIZE: i32 = 4096;

#[derive(Debug, Clone, Deserialize)]
pub struct Idefics3ImageProcessor {
    do_convert_rgb: bool,
    do_resize: bool,
    size: Option<HashMap<String, i32>>,
    do_image_splitting: bool,
    image_mean: Option<Vec<f32>>,
    image_std: Option<Vec<f32>>,

    #[serde(default = "default_max_image_size")]
    max_image_size: Option<HashMap<String, i32>>,
    do_rescale: bool,
    rescale_factor: f32,
    do_pad: bool,
    do_normalize: bool,
}

fn default_max_image_size() -> Option<HashMap<String, i32>> {
    Some(HashMap::from([("longest_edge".to_string(), 364)]))
}

impl Idefics3ImageProcessor {
    pub fn new(
        do_convert_rgb: bool,
        do_resize: bool,
        size: Option<HashMap<String, i32>>,
        do_image_splitting: bool,
        max_image_size: Option<HashMap<String, i32>>,
        do_rescale: bool,
        rescale_factor: f32,
        do_pad: bool,
        do_normalize: bool,
        image_mean: Option<Vec<f32>>,
        image_std: Option<Vec<f32>>,
    ) -> Self {
        let max_image_size =
            max_image_size.unwrap_or(HashMap::from([("longest_edge".to_string(), 364)]));
        let image_mean = image_mean.unwrap_or(vec![0.5, 0.5, 0.5]);
        let image_std = image_std.unwrap_or(vec![0.5, 0.5, 0.5]);
        Self {
            do_convert_rgb,
            do_resize,
            size,
            do_image_splitting,
            image_mean: Some(image_mean),
            image_std: Some(image_std),
            max_image_size: Some(max_image_size),
            do_rescale,
            rescale_factor,
            do_pad,
            do_normalize,
        }
    }

    pub fn from_pretrained(model_id: &str) -> Result<Self, anyhow::Error> {
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
        let config_file = repo.get("preprocessor_config.json").unwrap();
        let processor: Idefics3ImageProcessor =
            serde_json::from_slice(&std::fs::read(config_file).unwrap()).unwrap();
        Ok(processor)
    }

    pub fn resize_for_vision_encoder(
        &self,
        image: &DynamicImage,
        vision_encoder_max_size: &i32,
        resample: FilterType,
    ) -> ndarray::Array3<u8> {
        let (width, height) = image.dimensions();
        let height = height as f32;
        let width = width as f32;
        let vision_encoder_max_size = *vision_encoder_max_size as f32;

        let aspect_ratio = width / height;
        let (new_height, new_width) = if width >= height {
            let width = (width / vision_encoder_max_size).ceil() * vision_encoder_max_size;
            let height = (width / aspect_ratio).floor();
            let height =
                ((height / vision_encoder_max_size).ceil() * vision_encoder_max_size).ceil();
            (height, width)
        } else {
            let height = (height / vision_encoder_max_size).ceil() * vision_encoder_max_size;
            let width = (height * aspect_ratio).floor();
            let width = (width / vision_encoder_max_size).ceil() * vision_encoder_max_size;
            (height, width)
        };
        self.resize(
            &image,
            HashMap::from([
                ("width".to_string(), new_width as i32),
                ("height".to_string(), new_height as i32),
            ]),
            resample,
        )
    }

    pub fn resize(
        &self,
        image: &DynamicImage,
        size: HashMap<String, i32>,
        resample: FilterType,
    ) -> ndarray::Array3<u8> {
        let resized_image = if size.contains_key("height") && size.contains_key("width") {
            image.resize_exact(
                size.get("width").cloned().unwrap() as u32,
                size.get("height").cloned().unwrap() as u32,
                resample,
            )
        } else {
            let size = get_resize_output_image_size(
                image.clone(),
                size.get("longest_edge").cloned().unwrap(),
            );
            image.resize_exact(size.1 as u32, size.0 as u32, resample)
        };
        let (width, height) = resized_image.dimensions();
        let resized_image_array = resized_image.to_rgb8().into_raw();
        let resized_image_array = ndarray::Array3::from_shape_vec(
            (height as usize, width as usize, 3),
            resized_image_array,
        )
        .unwrap();
        resized_image_array
    }

    pub fn split_image(
        &self,
        image: &DynamicImage,
        max_image_size: &HashMap<String, i32>,
        resample: FilterType,
    ) -> (Vec<ndarray::Array3<u8>>, i32, i32) {
        let (width, height) = image.dimensions();
        let max_size = max_image_size.get("longest_edge").unwrap_or(&364);
        let max_height = *max_size;
        let max_width = *max_size;

        let mut frames = Vec::new();
        let (num_splits_h, num_splits_w) = if height > max_height as u32 || width > max_width as u32
        {
            // Calculate the number of splits
            let num_splits_h = (height as f32 / max_height as f32).ceil() as i32;
            let num_splits_w = (width as f32 / max_width as f32).ceil() as i32;

            // Calculate optimal dimensions for sub-images
            let optimal_height = (height as f32 / num_splits_h as f32).ceil() as u32;
            let optimal_width = (width as f32 / num_splits_w as f32).ceil() as u32;

            // Iterate through each row and column
            for r in 0..num_splits_h {
                for c in 0..num_splits_w {
                    // Calculate crop coordinates
                    let start_x = (c as u32) * optimal_width;
                    let start_y = (r as u32) * optimal_height;
                    let end_x = std::cmp::min(start_x + optimal_width, width);
                    let end_y = std::cmp::min(start_y + optimal_height, height);

                    // Crop the image
                    let cropped_image =
                        image.crop_imm(start_x, start_y, end_x - start_x, end_y - start_y);
                    frames.push(cropped_image);
                }
            }

            // For the global image at the end, we resize it to match the max_image_size, for cpu memory efficiency
            let global_image_height = max_height as u32;
            let global_image_width = max_width as u32;
            let global_image = if height != global_image_height || width != global_image_width {
                let size = HashMap::from([
                    ("height".to_string(), global_image_height as i32),
                    ("width".to_string(), global_image_width as i32),
                ]);
                let resized = self.resize(image, size, resample);
                DynamicImage::ImageRgb8(
                    RgbImage::from_raw(
                        resized.shape()[1] as u32,
                        resized.shape()[0] as u32,
                        resized.into_raw_vec_and_offset().0,
                    )
                    .unwrap(),
                )
            } else {
                image.clone()
            };
            frames.push(global_image);

            (num_splits_h, num_splits_w)
        } else {
            // If image is smaller than max_size, just add it as is
            frames.push(image.clone());
            (0, 0)
        };

        let frames = frames
            .iter()
            .map(|frame| {
                let image = frame.to_rgb8().into_raw();
                let image = ndarray::Array3::from_shape_vec(
                    (frame.height() as usize, frame.width() as usize, 3),
                    image,
                )
                .unwrap();
                image
            })
            .collect::<Vec<_>>();

        (frames, num_splits_h, num_splits_w)
    }

    pub fn rescale(
        &self,
        image: &ndarray::Array3<u8>,
        rescale_factor: f32,
    ) -> ndarray::Array3<f32> {
        let image = image.to_owned();
        let image = image.map(|x| *x as f32 * rescale_factor);
        image
    }

    fn pad_image(
        &self,
        image: &ndarray::Array3<f32>,
        output_size: (i32, i32),
        constant_values: f32,
        data_format: &str,
    ) -> (ndarray::Array3<f32>, ndarray::Array2<i32>) {
        let (input_height, input_width, _) = image.dim();
        let (output_height, output_width) = output_size;

        // Create padded image with zeros
        let mut padded_image = if data_format == "channels_first" {
            ndarray::Array3::zeros((3, output_height as usize, output_width as usize))
        } else {
            ndarray::Array3::zeros((output_height as usize, output_width as usize, 3))
        };

        // Create pixel attention mask
        let mut pixel_mask =
            ndarray::Array2::zeros((output_height as usize, output_width as usize));

        // Copy the original image into the padded image
        if data_format == "channels_first" {
            let transposed_image = image.to_owned().permuted_axes([2, 0, 1]);
            padded_image
                .slice_mut(s![.., ..input_height, ..input_width])
                .assign(&transposed_image);
            pixel_mask
                .slice_mut(s![..input_height, ..input_width])
                .fill(1);
        } else {
            padded_image
                .slice_mut(s![..input_height, ..input_width, ..])
                .assign(&image);
            pixel_mask
                .slice_mut(s![..input_height, ..input_width])
                .fill(1);
        }

        (padded_image, pixel_mask)
    }

    pub fn pad(
        &self,
        images: &[Vec<ndarray::Array3<f32>>],
        constant_values: f32,
        return_pixel_mask: bool,
        data_format: &str,
    ) -> (Vec<ndarray::Array3<f32>>, Option<Vec<ndarray::Array2<i32>>>) {
        // Get max dimensions across all images
        let mut max_height = 0;
        let mut max_width = 0;
        let mut max_num_images = 0;

        for batch in images {
            max_num_images = std::cmp::max(max_num_images, batch.len());
            for image in batch {
                let (height, width, _) = image.dim();
                println!("Image Shape: {:?}", image.shape());
                max_height = std::cmp::max(max_height, height);
                max_width = std::cmp::max(max_width, width);
            }
        }

        let output_size = (max_height as i32, max_width as i32);
        let batch_size = images.len();

        // Create empty padded images and masks
        let mut padded_images = vec![
            vec![
                if data_format == "channels_first" {
                    ndarray::Array3::zeros((3, max_height, max_width))
                } else {
                    ndarray::Array3::zeros((max_height, max_width, 3))
                };
                max_num_images
            ];
            batch_size
        ];

        let mut padded_masks = if return_pixel_mask {
            Some(vec![
                vec![
                    ndarray::Array2::zeros((max_height, max_width));
                    max_num_images
                ];
                batch_size
            ])
        } else {
            None
        };

        // Pad each image
        for (batch_idx, batch) in images.iter().enumerate() {
            for (sample_idx, image) in batch.iter().enumerate() {
                let (padded_image, pixel_mask) =
                    self.pad_image(image, output_size, constant_values, data_format);
                padded_images[batch_idx][sample_idx] = padded_image;
                if let Some(ref mut masks) = padded_masks {
                    masks[batch_idx][sample_idx] = pixel_mask;
                }
            }
        }

        (
            padded_images.into_iter().flatten().collect(),
            padded_masks.map(|masks| masks.into_iter().flatten().collect()),
        )
    }

    fn _preprocess_one_image(
        &self,
        image: &DynamicImage,
    ) -> Result<(ndarray::Array4<f32>, Option<ndarray::Array3<i32>>, i32, i32), anyhow::Error> {
        // Step 1: Initial resize
        let resized_image = self.resize(image, self.size.clone().unwrap(), FilterType::Lanczos3);

        // Convert back to DynamicImage for further processing
        let resized_dynamic_image = DynamicImage::ImageRgb8(
            RgbImage::from_raw(
                resized_image.shape()[1] as u32,
                resized_image.shape()[0] as u32,
                resized_image.into_raw_vec_and_offset().0,
            )
            .ok_or_else(|| anyhow::anyhow!("Failed to convert resized image to DynamicImage"))?,
        );

        // Step 2: Resize for vision encoder
        let vision_encoder_image = self.resize_for_vision_encoder(
            &resized_dynamic_image,
            &self.max_image_size.clone().unwrap()["longest_edge"],
            FilterType::Lanczos3,
        );

        let vision_encoder_image = DynamicImage::ImageRgb8(
            RgbImage::from_raw(
                vision_encoder_image.shape()[1] as u32,
                vision_encoder_image.shape()[0] as u32,
                vision_encoder_image.into_raw_vec_and_offset().0,
            )
            .ok_or_else(|| anyhow::anyhow!("Failed to convert resized image to DynamicImage"))?,
        );

        // Step 3: Split image if needed
        let (frames, n_rows, n_cols) = if self.do_image_splitting {
            self.split_image(
                &vision_encoder_image,
                &self.max_image_size.clone().unwrap(),
                FilterType::Lanczos3,
            )
        } else {
            let frame = vision_encoder_image.to_rgb8().into_raw();
            let frame = ndarray::Array3::from_shape_vec(
                (
                    vision_encoder_image.dimensions().0 as usize,
                    vision_encoder_image.dimensions().1 as usize,
                    3,
                ),
                frame,
            )
            .unwrap();
            (vec![frame], 0, 0)
        };

        // Step 4: Rescale frames
        let rescale_image_frames: Vec<ndarray::Array3<f32>> = if self.do_rescale {
            frames
                .iter()
                .map(|frame| self.rescale(frame, self.rescale_factor))
                .collect()
        } else {
            frames
                .iter()
                .map(|frame| frame.map(|x| *x as f32))
                .collect()
        };

        // Step 5: Normalize frames
        let normalized_frames = if self.do_normalize {
            let image_mean = self
                .image_mean
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Missing image_mean"))?;
            let image_std = self
                .image_std
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Missing image_std"))?;
            let image_mean = ndarray::Array3::from_shape_vec((1, 1, image_mean.len()), image_mean)?;
            let image_std = ndarray::Array3::from_shape_vec((1, 1, image_std.len()), image_std)?;

            rescale_image_frames
                .iter()
                .map(|frame| normalize(frame, &image_mean, &image_std))
                .collect()
        } else {
            rescale_image_frames
        };

        // Step 6: Pad and stack frames
        let (padded_images, padded_masks) = if self.do_pad {
            self.pad(&[normalized_frames], 0.0, true, "channels_first")
        } else {
            (normalized_frames, None)
        };

        // Stack frames into a single batch
        let image_views: Vec<_> = padded_images.iter().map(|arr| arr.view()).collect();
        let padded_images_concatenated = ndarray::stack(ndarray::Axis(0), &image_views)?;

        // Stack masks if they exist
        let padded_masks_concatenated = if let Some(masks) = padded_masks {
            let mask_views: Vec<_> = masks.iter().map(|arr| arr.view()).collect();
            Some(ndarray::stack(ndarray::Axis(0), &mask_views)?)
        } else {
            None
        };

        Ok((
            padded_images_concatenated,
            padded_masks_concatenated,
            n_rows,
            n_cols,
        ))
    }

    pub fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> Result<(ndarray::Array4<f32>, Option<ndarray::Array3<i32>>, i32, i32), anyhow::Error> {
        let mut preprocessed_images = Vec::new();
        let mut preprocessed_masks = Vec::new();
        let mut n_rows = 0;
        let mut n_cols = 0;
        for image in images {
            let (padded_images, padded_masks, rows, cols) = self._preprocess_one_image(image)?;
            preprocessed_images.push(padded_images);
            if let Some(mask) = padded_masks {
                preprocessed_masks.push(mask);
            }
            n_rows = rows;
            n_cols = cols;
        }
        let image_views: Vec<_> = preprocessed_images.iter().map(|arr| arr.view()).collect();
        let preprocessed_images = ndarray::concatenate(ndarray::Axis(0), &image_views)?;
        let preprocessed_masks = if !preprocessed_masks.is_empty() {
            let mask_views: Vec<_> = preprocessed_masks.iter().map(|arr| arr.view()).collect();
            Some(ndarray::concatenate(ndarray::Axis(0), &mask_views)?)
        } else {
            None
        };
        Ok((preprocessed_images, preprocessed_masks, n_rows, n_cols))
    }
}

pub struct Idefics3Processor {
    image_processor: Idefics3ImageProcessor,
    tokenizer: Tokenizer,
    regex: Regex,
    fake_image_token: AddedToken,
    image_token: AddedToken,
    end_of_utterance_token: AddedToken,
    global_image_tag: AddedToken,
}

impl Idefics3Processor {
    pub fn from_pretrained(model_id: &str) -> anyhow::Result<Self> {
        let image_processor = Idefics3ImageProcessor::from_pretrained(model_id)?;
        let mut tokenizer = Tokenizer::from_pretrained(model_id, None)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let fake_image_token = AddedToken::from("<fake_token_around_image>", true);
        let image_token = AddedToken::from("<image>", true);
        let end_of_utterance_token = AddedToken::from("<end_of_utterance>", true);
        let global_image_tag = AddedToken::from("<global-img>", true);
        tokenizer.add_special_tokens(&[
            fake_image_token.clone(),
            image_token.clone(),
            end_of_utterance_token.clone(),
            global_image_tag.clone(),
        ]);

        let regex = Regex::new(r"(\n?<global-img>\n?|<row_\d+_col_\d+>\n?)+").unwrap();
        Ok(Idefics3Processor {
            image_processor,
            tokenizer,
            regex,
            fake_image_token,
            image_token,
            end_of_utterance_token,
            global_image_tag,
        })
    }

    pub fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> anyhow::Result<(
        ndarray::Array2<i64>,
        ndarray::Array2<i64>,
        ndarray::Array4<f32>,
        Option<ndarray::Array3<i32>>,
    )> {
        let (preprocessed_images, preprocessed_masks, n_rows, n_cols) =
            self.image_processor.preprocess(images)?;
        println!("N rows: {:?}, N cols: {:?}", n_rows, n_cols);
        let image_prompt = _prompt_split_image(
            169,
            n_rows,
            n_cols,
            &self.fake_image_token,
            &self.image_token,
            &self.global_image_tag,
        );

        let prompt = "<|im_start|>user\n<image>Describe the image.<end_of_utterance>";

        // in the prompt replace the image_token with the image_prompt
        let prompt = prompt.replace(&self.image_token.content, &image_prompt);

        let encodings = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        let input_ids = Array2::from_shape_vec(
            (1, encodings.get_ids().len()),
            encodings.get_ids().iter().map(|x| *x as i64).collect(),
        )?;
        let attention_mask = Array2::from_shape_vec(
            (1, encodings.get_attention_mask().len()),
            encodings
                .get_attention_mask()
                .iter()
                .map(|x| *x as i64)
                .collect(),
        )?;

        // Count occurrences of token ID 49190 in the first row
        let token_count = input_ids.row(0).iter().filter(|&&x| x == 49190).count();

        Ok((
            input_ids,
            attention_mask,
            preprocessed_images,
            preprocessed_masks,
        ))
    }
}

fn _prompt_split_image(
    image_seq_len: i32,
    image_rows: i32,
    image_cols: i32,
    fake_token_around_image: &AddedToken,
    image_token: &AddedToken,
    global_img_token: &AddedToken,
) -> String {
    let mut text_split_images = String::new();
    for n_h in 0..image_rows {
        for n_w in 0..image_cols {
            text_split_images.push_str(&format!(
                "{}<row_{}_col_{}>{}",
                fake_token_around_image.content,
                n_h + 1,
                n_w + 1,
                image_token.content.repeat(image_seq_len as usize),
            ));
        }
        text_split_images.push_str("\n");
    }
    text_split_images.push_str(&format!(
        "\n{}{}{}{}",
        fake_token_around_image.content,
        global_img_token.content,
        image_token.content.repeat(image_seq_len as usize),
        fake_token_around_image.content
    ));
    text_split_images
}

fn get_resize_output_image_size(image: DynamicImage, resolution_max_side: i32) -> (i32, i32) {
    let (width, height) = image.dimensions();
    let (new_height, new_width) =
        _resize_output_size_rescale_to_max_len(height as i32, width as i32, 1, resolution_max_side);
    let (new_height, new_width) =
        _resize_output_size_scale_below_upper_bound(new_height, new_width, Some(MAX_IMAGE_SIZE));
    (new_height as i32, new_width as i32)
}

fn _resize_output_size_rescale_to_max_len(
    height: i32,
    width: i32,
    min_len: i32,
    max_len: i32,
) -> (i32, i32) {
    let max_len = if max_len == 0 {
        std::cmp::max(height, width)
    } else {
        max_len
    };
    let aspect_ratio = width as f32 / height as f32;
    let (new_width, new_height) = if width >= height {
        let new_width = max_len;
        let new_height = (new_width as f32 / aspect_ratio) as i32;
        if new_height % 2 != 0 {
            (new_width, new_height + 1)
        } else {
            (new_width, new_height)
        }
    } else {
        let new_height = max_len;
        let new_width = (new_height as f32 * aspect_ratio) as i32;
        if new_width % 2 != 0 {
            (new_width + 1, new_height)
        } else {
            (new_width, new_height)
        }
    };

    // Avoid resizing to a size smaller than min_len
    let new_height = std::cmp::max(new_height, min_len);
    let new_width = std::cmp::max(new_width, min_len);
    (new_height, new_width)
}

fn _resize_output_size_scale_below_upper_bound(
    height: i32,
    width: i32,
    max_len: Option<i32>,
) -> (i32, i32) {
    let max_len = max_len.unwrap_or_else(|| std::cmp::max(height, width));
    let aspect_ratio = width as f32 / height as f32;

    let (new_width, new_height) = if width >= height && width > max_len {
        let new_width = max_len;
        let new_height = (new_width as f32 / aspect_ratio) as i32;
        (new_width, new_height)
    } else if height > width && height > max_len {
        let new_height = max_len;
        let new_width = (new_height as f32 * aspect_ratio) as i32;
        (new_width, new_height)
    } else {
        (width, height)
    };

    // Avoid resizing to a size smaller than 1
    let new_height = std::cmp::max(new_height, 1);
    let new_width = std::cmp::max(new_width, 1);
    (new_height, new_width)
}

fn normalize(
    image: &ndarray::Array3<f32>,
    mean: &ndarray::Array3<f32>,
    std: &ndarray::Array3<f32>,
) -> ndarray::Array3<f32> {
    let normalized_image = (image - mean) / std;

    normalized_image
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::sync::Api;
    use hf_hub::{Repo, RepoType};
    use image::RgbImage;

    #[test]
    fn image_resize_test() {
        let image = image::open("/home/akshay/projects/EmbedAnything/test.jpg").unwrap();
        let image_array = image.to_rgb8().into_raw();

        let api = Api::new().unwrap();
        let repo = api.repo(Repo::new(
            "onnx-community/colSmol-256M-ONNX".to_string(),
            RepoType::Model,
        ));
        let config_file = repo.get("preprocessor_config.json").unwrap();
        let processor: Idefics3ImageProcessor =
            serde_json::from_slice(&std::fs::read(config_file).unwrap()).unwrap();
        println!("{:?}", processor);
        let resized_image = processor.resize(
            &image,
            processor.size.clone().unwrap(),
            FilterType::Lanczos3,
        );
        // println!("Resized Image: {:?}", resized_image.into_raw_vec_and_offset().0.len());
        // println!("Resized Image: {:?}", resized_image);

        let resized_dynamic_image = DynamicImage::ImageRgb8(
            RgbImage::from_raw(
                resized_image.shape()[1] as u32,
                resized_image.shape()[0] as u32,
                resized_image.into_raw_vec_and_offset().0,
            )
            .unwrap(),
        );
        println!(
            "Resized Dynamic Image: {:?}",
            resized_dynamic_image.dimensions()
        );
        let resized_image = processor.resize_for_vision_encoder(
            &resized_dynamic_image,
            &processor.max_image_size.clone().unwrap()["longest_edge"],
            FilterType::Lanczos3,
        );
        println!("Resized Image: {:?}", resized_image.shape());

        let (frames, num_splits_h, num_splits_w) = processor.split_image(
            &resized_dynamic_image,
            &processor.max_image_size.clone().unwrap(),
            FilterType::Lanczos3,
        );
        println!("Frames: {:?}", frames.len());
        println!(
            "Num Splits H: {:?}, Num Splits W: {:?}",
            num_splits_h, num_splits_w
        );

        let rescale_image_frames = frames
            .iter()
            .map(|frame| {
                let frame = frame.to_owned();
                processor.rescale(&frame, processor.rescale_factor)
            })
            .collect::<Vec<_>>();
        // println!("Rescale Image: {:?}", rescale_image);

        let image_mean = processor.image_mean.clone().unwrap();
        let image_std = processor.image_std.clone().unwrap();
        let image_mean =
            ndarray::Array3::from_shape_vec((1, 1, image_mean.len()), image_mean).unwrap();
        let image_std =
            ndarray::Array3::from_shape_vec((1, 1, image_std.len()), image_std).unwrap();
        let normalized_frames = rescale_image_frames
            .iter()
            .map(|frame| {
                let frame = frame.to_owned();
                normalize(&frame, &image_mean, &image_std)
            })
            .collect::<Vec<_>>();

        // println!("Normalized Frames: {:?}", normalized_frames.len());
        let (padded_images, padded_masks) =
            processor.pad(&[normalized_frames], 0.0, true, "channels_first");
        let image_views: Vec<_> = padded_images.iter().map(|arr| arr.view()).collect();
        let padded_images_concatenated = ndarray::stack(ndarray::Axis(0), &image_views).unwrap();
        println!(
            "Padded Images Concatenated: {:?}",
            padded_images_concatenated.shape()
        );
        if let Some(masks) = padded_masks {
            let mask_views: Vec<_> = masks.iter().map(|arr| arr.view()).collect();
            let padded_masks_concatenated = ndarray::stack(ndarray::Axis(0), &mask_views);
            println!("Padded Masks Concatenated: {:?}", padded_masks_concatenated);
        }
    }

    #[test]
    fn test_idefics3_processor() {
        let image = image::open("/home/akshay/projects/EmbedAnything/test.jpg").unwrap();
        let processor =
            Idefics3Processor::from_pretrained("onnx-community/colSmol-256M-ONNX").unwrap();
        processor.preprocess(&[image]).unwrap();
    }
}
