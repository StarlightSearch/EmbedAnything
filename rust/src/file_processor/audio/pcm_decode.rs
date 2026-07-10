#[cfg(feature = "audio")]
pub mod audio_processing {
    use symphonia::core::audio::GenericAudioBufferRef;
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    fn append_first_channel(buf: GenericAudioBufferRef<'_>, pcm_data: &mut Vec<f32>) {
        let mut planes = vec![Vec::<f32>::new()];
        buf.copy_to_vecs_planar(&mut planes);
        if let Some(channel) = planes.first() {
            pcm_data.extend(channel);
        }
    }

    pub(crate) fn pcm_decode<P: AsRef<std::path::Path>>(
        path: P,
    ) -> anyhow::Result<(Vec<f32>, u32)> {
        let src = std::fs::File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(src), Default::default());
        let hint = Hint::new();
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let mut format = symphonia::default::get_probe().probe(&hint, mss, fmt_opts, meta_opts)?;

        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| anyhow::anyhow!("no supported audio tracks"))?;

        let dec_opts: AudioDecoderOptions = Default::default();
        let audio_params = track
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .ok_or_else(|| anyhow::anyhow!("missing audio codec parameters"))?;

        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio_params, &dec_opts)
            .map_err(|_| anyhow::anyhow!("unsupported codec"))?;

        let track_id = track.id;
        let sample_rate = audio_params.sample_rate.unwrap_or(0);
        let mut pcm_data = Vec::new();

        while let Some(packet) = format.next_packet()? {
            while !format.metadata().is_latest() {
                format.metadata().pop();
            }

            if packet.track_id != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(buf) => append_first_channel(buf, &mut pcm_data),
                Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
                Err(symphonia::core::errors::Error::IoError(_)) => continue,
                Err(err) => return Err(err.into()),
            }
        }

        Ok((pcm_data, sample_rate))
    }
}
