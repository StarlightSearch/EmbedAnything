use embed_anything::{
    chunkers::{cumulative::CumulativeChunker, statistical::StatisticalChunker},
    embeddings::local::jina::JinaEmbeder,
    text_loader::TextLoader,
};
use text_cleaner::clean::Clean;
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::Tokenizer;

fn main() {
    let _splitter = TextSplitter::new(
        ChunkConfig::new(100)
            .with_sizer(Tokenizer::from_pretrained("bert-base-cased", None).unwrap()),
    );
    let text = TextLoader::extract_text("test_files/linear.pdf").unwrap();

    // get first 1000 character
    let text = text
        .chars()
        .take(60000)
        .collect::<String>()
        .remove_leading_spaces()
        .remove_empty_lines()
        .remove_trailing_spaces();

    // let text = "The Bank of Elarian nestled in the heart of the bustling city of Elarian, the Bank of Elarian stands as a beacon of financial stability and innovative banking. Founded over a century ago, this bank has grown from a modest community institution into one of the city's most trusted financial centers. Known for its majestic architecture, the building's facade is a blend of classical and modern design, featuring high pillars and sleek glass panels that reflect the city's skyline.
    // Inside, the Bank of Elarian is equipped with state-of-the-art technology, offering customers a seamless banking experience. From advanced ATMs to virtual financial advisors, the bank has embraced digital transformation while maintaining a personal touch in customer service. Its range of services includes traditional savings and checking accounts, investment advisory, and a highly regarded private banking division catering to high net worth individuals.
    // The bank is also committed to community development, funding various local projects, and supporting small businesses, reflecting its deep roots in the Elarian community. Sapore di Mare Restaurant - Just a few blocks from the Bank of Elarian, Sapore di Mare offers a culinary journey with its exquisite seafood and Mediterranean cuisine. Its cozy, nautical-themed interior, complete with wooden accents and subtle lighting, creates a warm and inviting atmosphere.
    // The restaurant's signature dish is the Fruit of the Sea platter, featuring the freshest catch from local fishermen, cooked to perfection with herbs and spices that highlight the natural flavors. The chef, a native of the Mediterranean coast, brings authentic recipes and a passion for seafood to the table, ensuring each dish is a masterpiece. Sapore di Mare is not just known for its food but also for its exceptional service. The staff go above and beyond to create a memorable dining experience, making it a popular destination for both locals and tourists. Elarian Freiseur - A short stroll from Sapore di Mare is Elarian Freiseur, a boutique hair salon known for its chic style and innovative hair treatments. This salon stands out with its modern design, featuring sleek chairs, ambient lighting, and an array of plants that add a touch of greenery and freshness.
    // The team at Elarian Freiseur is comprised of highly skilled stylists and colorists who are experts in the latest hair trends. They offer personalized consultations to each client, ensuring a customized experience that meets individual style preferences. From classic cuts to avant-garde hair coloring, the salon is a hub for those seeking a transformative hair experience.
    // Elarian Freiseur also places a high emphasis on using eco-friendly and sustainable hair products, aligning with the city's growing environmental consciousness. This commitment to quality and sustainability has earned it a loyal clientele who appreciate the salon's dedication to both style and the environment.";

    // let chunks: Vec<&str> = splitter.chunks(&text).collect();

    // for chunk in chunks {
    //     println!("{:?}", chunk);
    // }

    // let encoder = JinaEmbeder::default();

    // let chunker = CumulativeChunker::new(encoder, splitter, 0.9);
    // chunker._chunk(&text);

    let chunker = StatisticalChunker {
        verbose: true,
        ..Default::default()
    };
    chunker._chunk(&text, 64);
}
