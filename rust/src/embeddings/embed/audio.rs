use crate::file_processor::audio::audio_processor::Segment;
use anyhow::Error;
use std::path::Path;

pub trait AudioDecoder {
    fn decode_audio(&mut self, audio_file: &Path) -> Result<Vec<Segment>, Error>;
}
