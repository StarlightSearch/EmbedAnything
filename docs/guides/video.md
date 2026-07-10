# Video Embeddings (Frame Sampling)

EmbedAnything supports video by sampling frames and embedding them with a vision model
(CLIP/SigLIP). This is opt-in via the `video` feature flag and requires the `ffmpeg`
CLI to be available on your system. If `ffmpeg` is not on `PATH`, set `FFMPEG_BIN`
to the full path of the executable.

## Recommended Config

`VideoEmbedConfig` controls how frames are sampled:

- `frame_step`: sample every Nth frame. Default `30`.
- `max_frames`: maximum frames per video. Default `300`.
- `batch_size`: frames per embedding batch. Default `32`.

Suggested starting point:

```python
from embed_anything import VideoEmbedConfig

config = VideoEmbedConfig(frame_step=30, max_frames=300, batch_size=16)
```

## Python Usage

```python
import embed_anything
from embed_anything import VideoEmbedConfig

model = embed_anything.EmbeddingModel.from_pretrained_hf(
    model_id="openai/clip-vit-base-patch16"
)

config = VideoEmbedConfig(frame_step=30, max_frames=200, batch_size=16)

data = embed_anything.embed_video_file("path/to/video.mp4", embedder=model, config=config)
```

## Build with Video Support

You must enable the `video` feature and have the `ffmpeg` CLI installed.

### macOS

```bash
brew install ffmpeg
cargo build --features video
# Python (maturin)
maturin develop --features "extension-module,video"
```

### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
cargo build --features video
# Python (maturin)
maturin develop --features "extension-module,video"
```

### Windows (prebuilt FFmpeg)

```powershell
1. Download a static build from https://www.gyan.dev/ffmpeg/builds/
2. Extract it and set:

```powershell
$env:FFMPEG_BIN = "C:\path\to\ffmpeg.exe"
```

Then build:

```powershell
cargo build --features video
# Python (maturin)
maturin develop --features "extension-module,video"
```
```

## Output Metadata

Each embedding includes:

- `video_path`: the source video file
- `frame_index`: the sampled frame index (0-based)
