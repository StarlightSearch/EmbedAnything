import os
from pathlib import Path

import embed_anything
from embed_anything import EmbedData, VideoEmbedConfig

# Load a vision model (CLIP/SigLIP) for frame embeddings
model = embed_anything.EmbeddingModel.from_pretrained_hf(
    model_id="openai/clip-vit-base-patch16"
)

# Sample every 30th frame (~1 fps for 30 fps videos), cap to 200 frames
config = VideoEmbedConfig(frame_step=30, max_frames=200, batch_size=16)

video_path = os.environ.get("VIDEO_PATH", "path/to/video.mp4")
if not Path(video_path).exists():
    raise FileNotFoundError(
        f"Video not found: {video_path}. Set VIDEO_PATH env var to a valid file."
    )

# Embed a single video
video_embeddings: list[EmbedData] = embed_anything.embed_video_file(
    video_path,
    embedder=model,
    config=config,
)
print(f"Embedded {len(video_embeddings)} frames from video.")

video_dir = os.environ.get("VIDEO_DIR")
if video_dir:
    dir_embeddings = embed_anything.embed_video_directory(
        video_dir,
        embedder=model,
        config=config,
    )
    if dir_embeddings is not None:
        print(f"Embedded {len(dir_embeddings)} total frames from directory.")
