#![cfg(feature = "video")]

use anyhow::{anyhow, Result};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{fs, path};

#[derive(Debug, Clone, Copy)]
pub enum VideoFrameFormat {
    Jpeg,
    Png,
}

impl VideoFrameFormat {
    fn extension(self) -> &'static str {
        match self {
            VideoFrameFormat::Jpeg => "jpg",
            VideoFrameFormat::Png => "png",
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub index: usize,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct VideoProcessor {
    frame_step: usize,
    max_frames: Option<usize>,
    output_format: VideoFrameFormat,
    ffmpeg_bin: Option<PathBuf>,
}

impl VideoProcessor {
    pub fn new(frame_step: usize) -> Self {
        Self {
            frame_step: frame_step.max(1),
            max_frames: None,
            output_format: VideoFrameFormat::Jpeg,
            ffmpeg_bin: None,
        }
    }

    pub fn with_max_frames(mut self, max_frames: usize) -> Self {
        self.max_frames = Some(max_frames);
        self
    }

    pub fn with_output_format(mut self, output_format: VideoFrameFormat) -> Self {
        self.output_format = output_format;
        self
    }

    pub fn with_ffmpeg_bin<P: AsRef<Path>>(mut self, ffmpeg_bin: P) -> Self {
        self.ffmpeg_bin = Some(ffmpeg_bin.as_ref().to_path_buf());
        self
    }

    fn resolve_ffmpeg_bin(&self) -> Result<PathBuf> {
        if let Some(bin) = &self.ffmpeg_bin {
            return Ok(bin.clone());
        }
        if let Ok(bin) = env::var("FFMPEG_BIN") {
            return Ok(PathBuf::from(bin));
        }
        Ok(PathBuf::from("ffmpeg"))
    }

    pub fn extract_frames_to_dir<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        video_path: P,
        output_dir: Q,
    ) -> Result<Vec<VideoFrame>> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir)?;

        let ffmpeg_bin = self.resolve_ffmpeg_bin()?;
        let frame_step = self.frame_step.max(1);
        let filter = format!("select=not(mod(n\\,{}))", frame_step);
        let output_pattern = output_dir.join(format!(
            "frame_%06d.{}",
            self.output_format.extension()
        ));

        let mut command = Command::new(ffmpeg_bin);
        command
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-i")
            .arg(video_path.as_ref())
            .arg("-vf")
            .arg(filter)
            .arg("-vsync")
            .arg("vfr");

        if let Some(max_frames) = self.max_frames {
            command.arg("-vframes").arg(max_frames.to_string());
        }

        let status = command.arg(output_pattern).status()?;
        if !status.success() {
            return Err(anyhow!("ffmpeg failed with exit code {:?}", status.code()));
        }

        let mut frame_paths = fs::read_dir(output_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().map(|t| t.is_file()).unwrap_or(false))
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case(self.output_format.extension()))
                    .unwrap_or(false)
            })
            .collect::<Vec<path::PathBuf>>();

        frame_paths.sort();

        if frame_paths.is_empty() {
            return Err(anyhow!("No frames extracted from video"));
        }

        let frames = frame_paths
            .into_iter()
            .enumerate()
            .map(|(index, path)| VideoFrame { index, path })
            .collect();

        Ok(frames)
    }

    pub fn extract_frames_to_temp_dir<P: AsRef<Path>>(
        &self,
        video_path: P,
    ) -> Result<(tempfile::TempDir, Vec<VideoFrame>)> {
        let temp_dir = tempfile::TempDir::new()?;
        let frames = self.extract_frames_to_dir(video_path, temp_dir.path())?;
        Ok((temp_dir, frames))
    }
}
