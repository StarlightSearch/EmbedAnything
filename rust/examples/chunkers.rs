use embed_anything::text_loader::TextLoader;
use embed_anything::chunkers::statistical::StatisticalChunker;
fn main() {
    let text = TextLoader::extract_text("/home/akshay/EmbedAnything/test_files/attention.pdf").unwrap();
    let chunker = StatisticalChunker{
        verbose: true,
        ..Default::default()
    };
    let chunks = chunker._chunk(&text, 32);

}