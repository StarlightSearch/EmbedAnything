use candle_core::{DType, Device, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};
use serde::Deserialize;
use std::collections::HashMap;
use tokenizers::{AddedToken, Tokenizer};

const MAX_IMAGE_SIZE: i32 = 4096;

type ProcessedImageResult = Result<(Vec<Tensor>, Option<Vec<Tensor>>, i32, i32), anyhow::Error>;

#[derive(Debug, Clone, Deserialize)]
pub struct Idefics3ImageProcessor {
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
    pub fn resize_for_vision_encoder(
        &self,
        image: &DynamicImage,
        vision_encoder_max_size: &i32,
        resample: FilterType,
        device: &Device,
    ) -> Result<Tensor, anyhow::Error> {
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
            image,
            HashMap::from([
                ("width".to_string(), new_width as i32),
                ("height".to_string(), new_height as i32),
            ]),
            resample,
            device,
        )
    }

    pub fn resize(
        &self,
        image: &DynamicImage,
        size: HashMap<String, i32>,
        resample: FilterType,
        device: &Device,
    ) -> Result<Tensor, anyhow::Error> {
        let resized_image = if size.contains_key("height") && size.contains_key("width") {
            image.resize_exact(
                size.get("width")
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("Missing width"))? as u32,
                size.get("height")
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("Missing height"))? as u32,
                resample,
            )
        } else {
            let size = get_resize_output_image_size(
                image.clone(),
                size.get("longest_edge")
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("Missing longest_edge"))?,
            );
            image.resize_exact(size.1 as u32, size.0 as u32, resample)
        };
        let (width, height) = resized_image.dimensions();
        let resized_image_array = resized_image.to_rgb8().into_raw();
        let resized_image_array = Tensor::from_vec(
            resized_image_array,
            (height as usize, width as usize, 3),
            device,
        )?;
        Ok(resized_image_array)
    }

    pub fn split_image(
        &self,
        image: &DynamicImage,
        max_image_size: &HashMap<String, i32>,
        resample: FilterType,
        device: &Device,
    ) -> Result<(Vec<Tensor>, i32, i32), anyhow::Error> {
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
                let resized = self.resize(image, size, resample, device)?;
                DynamicImage::ImageRgb8(
                    RgbImage::from_raw(
                        resized.dims()[1] as u32,
                        resized.dims()[0] as u32,
                        resized.flatten_all()?.to_vec1::<u8>()?,
                    )
                    .ok_or_else(|| {
                        anyhow::anyhow!("Failed to convert resized image to DynamicImage")
                    })?,
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
            .map(|frame| -> Result<Tensor, anyhow::Error> {
                let image = frame.to_rgb8().into_raw();
                Ok(Tensor::from_vec(
                    image,
                    (frame.height() as usize, frame.width() as usize, 3),
                    device,
                )?)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((frames, num_splits_h, num_splits_w))
    }

    fn rescale(&self, image: &Tensor, rescale_factor: f32) -> Result<Tensor, anyhow::Error> {
        let image = (image.to_dtype(DType::F32)? * rescale_factor as f64)?;
        Ok(image)
    }

    fn pad_image(
        &self,
        image: &Tensor,
        output_size: (i32, i32),
        data_format: &str,
        device: &Device,
    ) -> Result<(Tensor, Tensor), anyhow::Error> {
        let (input_height, input_width, _) = image.dims3()?;
        let (output_height, output_width) = output_size;

        // Create padded image with zeros
        let mut padded_image = if data_format == "channels_first" {
            Tensor::zeros(
                (3, output_height as usize, output_width as usize),
                DType::F32,
                device,
            )?
        } else {
            Tensor::zeros(
                (output_height as usize, output_width as usize, 3),
                DType::F32,
                device,
            )?
        };

        // Create pixel attention mask
        let mut pixel_mask = Tensor::zeros(
            (output_height as usize, output_width as usize),
            DType::I64,
            device,
        )?;

        // Copy the original image into the padded image
        if data_format == "channels_first" {
            let transposed_image = image.to_owned().permute([2, 0, 1])?;
            padded_image = padded_image
                .slice_assign(&[0..3, 0..input_height, 0..input_width], &transposed_image)?;
            pixel_mask = pixel_mask.slice_assign(
                &[0..input_height, 0..input_width],
                &Tensor::ones((input_height, input_width), DType::I64, device)?,
            )?;
        } else {
            padded_image =
                padded_image.slice_assign(&[0..3, 0..input_height, 0..input_width], image)?;
            pixel_mask = pixel_mask.slice_assign(
                &[0..input_height, 0..input_width],
                &Tensor::ones((input_height, input_width), DType::I64, device)?,
            )?;
        }

        Ok((padded_image, pixel_mask))
    }

    pub fn pad(
        &self,
        images: &[Vec<Tensor>],
        return_pixel_mask: bool,
        data_format: &str,
        device: &Device,
    ) -> Result<(Vec<Tensor>, Option<Vec<Tensor>>), anyhow::Error> {
        // Get max dimensions across all images
        let mut max_height = 0;
        let mut max_width = 0;
        let mut max_num_images = 0;

        for batch in images {
            max_num_images = std::cmp::max(max_num_images, batch.len());
            for image in batch {
                let (height, width, _) = image
                    .dims3()
                    .map_err(|e| anyhow::anyhow!("Error getting dimensions: {}", e))?;
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
                    Tensor::zeros((3, max_height, max_width), DType::F32, device)?
                } else {
                    Tensor::zeros((max_height, max_width, 3), DType::F32, device)?
                };
                max_num_images
            ];
            batch_size
        ];

        let mut padded_masks = if return_pixel_mask {
            Some(vec![
                vec![
                    Tensor::zeros(
                        (max_height, max_width),
                        DType::I64,
                        device
                    )?;
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
                    self.pad_image(image, output_size, data_format, device)?;
                padded_images[batch_idx][sample_idx] = padded_image;
                if let Some(ref mut masks) = padded_masks {
                    masks[batch_idx][sample_idx] = pixel_mask;
                }
            }
        }

        Ok((
            padded_images.into_iter().flatten().collect(),
            padded_masks.map(|masks| masks.into_iter().flatten().collect()),
        ))
    }

    fn _preprocess_one_image(&self, image: &DynamicImage, device: &Device) -> ProcessedImageResult {
        // Step 1: Initial resize
        let resized_image = self.resize(
            image,
            self.size
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Missing size"))?,
            FilterType::Lanczos3,
            device,
        )?;

        // Convert back to DynamicImage for further processing
        let resized_dynamic_image = DynamicImage::ImageRgb8(
            RgbImage::from_raw(
                resized_image.dims()[1] as u32,
                resized_image.dims()[0] as u32,
                resized_image.flatten_all()?.to_vec1::<u8>()?,
            )
            .ok_or_else(|| anyhow::anyhow!("Failed to convert resized image to DynamicImage"))?,
        );

        // Step 2: Resize for vision encoder
        let vision_encoder_image = self.resize_for_vision_encoder(
            &resized_dynamic_image,
            &self
                .max_image_size
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Missing max_image_size"))?["longest_edge"],
            FilterType::Lanczos3,
            device,
        )?;

        let vision_encoder_image = DynamicImage::ImageRgb8(
            RgbImage::from_raw(
                vision_encoder_image.dims()[1] as u32,
                vision_encoder_image.dims()[0] as u32,
                vision_encoder_image.flatten_all()?.to_vec1::<u8>()?,
            )
            .ok_or_else(|| anyhow::anyhow!("Failed to create RgbImage from raw data"))?,
        );

        // Step 3: Split image if needed
        let (frames, n_rows, n_cols) = if self.do_image_splitting {
            self.split_image(
                &vision_encoder_image,
                &self
                    .max_image_size
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("Missing max_image_size"))?,
                FilterType::Lanczos3,
                device,
            )?
        } else {
            let frame = vision_encoder_image.to_rgb8().into_raw();
            let frame = Tensor::from_vec(
                frame,
                (
                    vision_encoder_image.height() as usize,
                    vision_encoder_image.width() as usize,
                    3,
                ),
                device,
            )?;
            (vec![frame], 1, 1)
        };

        // Step 4 & 5: Rescale and normalize frames
        let normalized_frames = if self.do_normalize {
            let image_mean = self
                .image_mean
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Missing image_mean"))?;
            let image_std = self
                .image_std
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Missing image_std"))?;
            let image_mean =
                Tensor::from_vec(image_mean.clone(), (1, 1, image_mean.len()), device)?;
            let image_std = Tensor::from_vec(image_std.clone(), (1, 1, image_std.len()), device)?;

            frames
                .iter()
                .map(|frame| {
                    // First rescale the frame
                    let rescaled_frame = if self.do_rescale {
                        self.rescale(frame, self.rescale_factor)?
                    } else {
                        frame.to_dtype(DType::F32)?
                    };
                    // Then normalize the rescaled frame
                    normalize(&rescaled_frame, &image_mean, &image_std)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // Only rescale without normalization
            frames
                .iter()
                .map(|frame| -> Result<Tensor, anyhow::Error> {
                    if self.do_rescale {
                        self.rescale(frame, self.rescale_factor)
                    } else {
                        Ok(frame.to_dtype(DType::F32)?)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Step 6: Pad and stack frames
        let (padded_images, padded_masks) = if self.do_pad {
            self.pad(&[normalized_frames], true, "channels_first", device)?
        } else {
            (normalized_frames, None)
        };

        Ok((padded_images, padded_masks, n_rows, n_cols))
    }

    pub fn preprocess(
        &self,
        images: &[DynamicImage],
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>, i32, i32), anyhow::Error> {
        let mut preprocessed_images = Vec::new();
        let mut preprocessed_masks = Vec::new();
        let mut n_rows = 0;
        let mut n_cols = 0;
        for image in images {
            let (padded_images, padded_masks, rows, cols) =
                self._preprocess_one_image(image, device)?;
            preprocessed_images.push(padded_images);
            if let Some(mask) = padded_masks {
                preprocessed_masks.push(mask);
            }
            n_rows = rows;
            n_cols = cols;
        }
        let max_num_images = preprocessed_images
            .iter()
            .map(|image| image.len())
            .max()
            .expect("No images found");
        let (max_height, max_width) = get_max_height_width(&preprocessed_images);
        let mut padded_image_list_full = vec![];
        let mut padded_mask_list_full = vec![];
        for _ in images {
            let (padded_image_list, padded_mask_list): (Vec<_>, Vec<_>) = (0..max_num_images)
                .map(|_| -> Result<(Tensor, Tensor), anyhow::Error> {
                    Ok((
                        empty_image(3, max_height as u32, max_width as u32, DType::F32, device)?,
                        Tensor::zeros((max_height, max_width), DType::I64, device)?,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .unzip();
            padded_image_list_full.push(padded_image_list);
            padded_mask_list_full.push(padded_mask_list);
        }
        for (i, image) in preprocessed_images.iter().enumerate() {
            padded_image_list_full[i][..image.len()].clone_from_slice(&image[..]);
        }

        let (padded_image_list_full, padded_mask_list_full): (Vec<_>, Vec<_>) =
            padded_image_list_full
                .iter()
                .zip(padded_mask_list_full.iter())
                .map(
                    |(image_list, mask_list)| -> Result<(Tensor, Tensor), anyhow::Error> {
                        Ok((Tensor::stack(image_list, 0)?, Tensor::stack(mask_list, 0)?))
                    },
                )
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .unzip();

        let padded_image_list_full = Tensor::stack(&padded_image_list_full, 0)?;
        let padded_mask_list_full = Tensor::stack(&padded_mask_list_full, 0)?;

        Ok((
            padded_image_list_full,
            Some(padded_mask_list_full),
            n_rows,
            n_cols,
        ))
    }

    pub fn from_pretrained(model_id: &str) -> Result<Self, anyhow::Error> {
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
        let config_file = repo.get("preprocessor_config.json")?;
        let processor: Idefics3ImageProcessor =
            serde_json::from_slice(&std::fs::read(config_file)?)
                .map_err(|e| anyhow::anyhow!("Failed to read preprocessor config: {}", e))?;
        Ok(processor)
    }
}

pub struct Idefics3Processor {
    image_processor: Idefics3ImageProcessor,
    tokenizer: Tokenizer,
    fake_image_token: AddedToken,
    image_token: AddedToken,
    global_image_tag: AddedToken,
    image_seq_len: i32,
}

impl Idefics3Processor {
    pub fn from_pretrained(model_id: &str) -> anyhow::Result<Self> {
        let image_processor = Idefics3ImageProcessor::from_pretrained(model_id)?;
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let processor_config_file = repo.get("processor_config.json")?;
        let processor_config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(processor_config_file)?)?;
        let image_seq_len = processor_config["image_seq_len"].as_u64().unwrap_or(64) as i32;
        let config_file = repo.get("tokenizer.json")?;
        let mut tokenizer = Tokenizer::from_file(config_file)
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

        Ok(Idefics3Processor {
            image_processor,
            tokenizer,
            fake_image_token,
            image_token,
            global_image_tag,
            image_seq_len,
        })
    }

    pub fn tokenize_batch(
        &self,
        prompts: Vec<&str>,
        device: &Device,
    ) -> Result<(Tensor, Tensor), anyhow::Error> {
        let new_prompts = prompts
            .iter()
            .map(|prompt| {
                let prompt = format!("Query: {} {} \n", prompt, "<end_of_utterance>".repeat(10));
                prompt
            })
            .collect::<Vec<_>>();

        let tokens = self
            .tokenizer
            .encode_batch(new_prompts, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input = Tensor::stack(&token_ids, 0)?;
        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        Ok((input, attention_mask))
    }

    pub fn preprocess(
        &self,
        images: &[DynamicImage],
        device: &Device,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        let (preprocessed_images, preprocessed_masks, n_rows, n_cols) =
            self.image_processor.preprocess(images, device)?;
        let image_prompt = _prompt_split_image(
            self.image_seq_len,
            n_rows,
            n_cols,
            &self.fake_image_token,
            &self.image_token,
            &self.global_image_tag,
        );

        let prompt = "<|im_start|>user\n<image>Describe the image.<end_of_utterance>";

        // in the prompt replace the image_token with the image_prompt
        let mut prompts = Vec::new();
        for _ in images {
            let prompt = prompt.replace(&self.image_token.content, &image_prompt);
            prompts.push(prompt);
        }

        let (input_ids, attention_mask) =
            self.tokenize_batch(prompts.iter().map(|x| x.as_str()).collect(), device)?;

        Ok((
            input_ids,
            attention_mask,
            preprocessed_images,
            preprocessed_masks,
        ))
    }
}

fn get_max_height_width(image_list: &[Vec<Tensor>]) -> (usize, usize) {
    let mut max_height = 0;
    let mut max_width = 0;
    for images in image_list {
        for image in images {
            let height = image.dims()[1];
            let width = image.dims()[2];
            max_height = std::cmp::max(max_height, height);
            max_width = std::cmp::max(max_width, width);
        }
    }
    (max_height, max_width)
}

fn empty_image(
    channels: usize,
    height: u32,
    width: u32,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, anyhow::Error> {
    Ok(Tensor::zeros(
        (channels, height as usize, width as usize),
        dtype,
        device,
    )?)
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
        text_split_images.push('\n');
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
    (new_height, new_width)
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

fn normalize(image: &Tensor, mean: &Tensor, std: &Tensor) -> Result<Tensor, anyhow::Error> {
    let normalized_image = image
        .to_dtype(DType::F32)?
        .broadcast_sub(mean)?
        .broadcast_div(std)?;
    Ok(normalized_image)
}
