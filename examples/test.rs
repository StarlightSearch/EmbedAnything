use embed_anything::text_loader::TextLoader;
use text_splitter::TextSplitter;

fn main(){
    let splitter = TextSplitter::new(1000);
    let text = TextLoader::extract_text("test_files/test.pdf").unwrap();

    let chunks:Vec<&str> = splitter.chunks(&text).collect();

    for chunk in chunks {
        println!("{:?}", chunk);
    }
}