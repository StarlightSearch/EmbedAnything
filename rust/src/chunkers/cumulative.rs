use crate::embeddings::{embed::TextEmbedder, local::jina::JinaEmbedder};
use candle_core::Tensor;
use text_splitter::{ChunkConfig, ChunkSizer, TextSplitter};
use tokenizers::Tokenizer;

pub struct CumulativeChunker<Sizer: ChunkSizer> {
    pub encoder: TextEmbedder,
    pub splitter: TextSplitter<Sizer>,
    pub score_threshold: f32,
    pub device: candle_core::Device,
}

impl Default for CumulativeChunker<Tokenizer> {
    fn default() -> Self {
        let splitter = TextSplitter::new(ChunkConfig::new(200).with_sizer(
            Tokenizer::from_pretrained("BEE-spoke-data/cl100k_base-mlm", None).unwrap(),
        ));
        let encoder = TextEmbedder::Jina(JinaEmbedder::default());
        let score_threshold = 0.9;
        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        Self {
            encoder,
            splitter,
            score_threshold,
            device,
        }
    }
}

impl<Sizer: ChunkSizer> CumulativeChunker<Sizer> {
    pub fn new(encoder: TextEmbedder, splitter: TextSplitter<Sizer>, score_threshold: f32) -> Self {
        Self {
            encoder,
            splitter,
            score_threshold,
            device: candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
        }
    }

    pub async fn _chunk(&self, text: &str) {
        let splits = self
            .splitter
            .chunks(text)
            .collect::<Vec<_>>()
            .into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>();

        let mut chunks: Vec<String> = Vec::new();
        let mut curr_chunk_idx = 0;
        let num_splits = splits.len();

        for idx in 0..num_splits {
            let curr_chunk_docs: String;
            let next_doc: String;
            //ensure there is a next document to compare with.
            if idx + 1 < num_splits {
                if idx == 0 {
                    // On the first iteration, compare the
                    // first document directly to the second.
                    curr_chunk_docs = splits[idx].to_string();
                } else {
                    // println!("Current_chunk_idx= {}, Split: {:?},idx: {}",curr_chunk_idx, splits[curr_chunk_idx..idx].to_vec(), idx);
                    let curr_chunk_docs_str = splits[curr_chunk_idx..idx + 1].join("\n");
                    curr_chunk_docs = curr_chunk_docs_str.to_string();
                }
                next_doc = splits[idx + 1].clone();

                let curr_chunk_docs_embed = self
                    .encoder
                    .embed(&[curr_chunk_docs], Some(32))
                    .await
                    .unwrap();

                let curr_chunk_docs_embed = curr_chunk_docs_embed
                    .into_iter()
                    .next()
                    .unwrap()
                    .to_dense()
                    .unwrap();

                let next_doc_embed = self
                    .encoder
                    .embed(&[next_doc.to_string()], Some(32))
                    .await
                    .unwrap();

                let next_doc_embed = next_doc_embed
                    .into_iter()
                    .next()
                    .unwrap()
                    .to_dense()
                    .unwrap();

                let curr_sim_score = self._cosine_similarity(curr_chunk_docs_embed, next_doc_embed);
                //decision to chunk based on similarity score.
                if curr_sim_score < self.score_threshold {
                    chunks.push(splits[curr_chunk_idx..idx + 1].join("\n"));
                    curr_chunk_idx = idx + 1; // update the start index for the next segment
                }
            }
        }

        if curr_chunk_idx < num_splits {
            let last_chunk = splits[curr_chunk_idx..].join("\n");
            chunks.push(last_chunk);
        }

        for chunk in chunks {
            println!("----------CHUNK ------\n{}", chunk);
        }
    }

    fn _cosine_similarity(&self, a: Vec<f32>, b: Vec<f32>) -> f32 {
        let embed_dim = a.len();

        // convert a and b to tensors
        let a_tensor = Tensor::from_vec(a, (1, embed_dim), &self.device).unwrap();
        let b_tensor = Tensor::from_vec(b, (1, embed_dim), &self.device).unwrap();

        // divide by norm of a and b
        let a_norm = a_tensor.sqr().unwrap().sum(1).unwrap().sqrt().unwrap();
        let b_norm = b_tensor.sqr().unwrap().sum(1).unwrap().sqrt().unwrap();
        let norm = (a_norm * b_norm)
            .unwrap()
            .get(0)
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();

        // calculate the dot product
        let dot_product = a_tensor
            .matmul(&b_tensor.transpose(0, 1).unwrap())
            .unwrap()
            .get(0)
            .unwrap()
            .get(0)
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();

        dot_product / norm
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_cumulative_chunker() {
        let text = "
        The Bank of Elarian
Nestled in the heart of the bustling city of Elarian, the Bank of Elarian stands as a beacon of financial stability and innovative banking. Founded over a century ago, this bank has grown from a modest community institution into one of the city's most trusted financial centers. Known for its majestic architecture, the building's facade is a blend of classical and modern design, featuring high pillars and sleek glass panels that reflect the city's skyline.
Inside, the Bank of Elarian is equipped with state-of-the-art technology, offering customers a seamless banking experience. From advanced ATMs to virtual financial advisors, the bank has embraced digital transformation while maintaining a personal touch in customer service. Its range of services includes traditional savings and checking accounts, investment advisory, and a highly regarded private banking division catering to high net worth individuals.
The bank is also committed to community development, funding various local projects, and supporting small businesses, reflecting its deep roots in the Elarian community.\n
Sapore di Mare Restaurant - Just a few blocks from the Bank of Elarian, Sapore di Mare offers a culinary journey with its exquisite seafood and Mediterranean cuisine. Its cozy, nautical-themed interior, complete with wooden accents and subtle lighting, creates a warm and inviting atmosphere.
The restaurant's signature dish is the Fruit of the Sea platter, featuring the freshest catch from local fishermen, cooked to perfection with herbs and spices that highlight the natural flavors. The chef, a native of the Mediterranean coast, brings authentic recipes and a passion for seafood to the table, ensuring each dish is a masterpiece.
Sapore di Mare is not just known for its food but also for its exceptional service. The staff go above and beyond to create a memorable dining experience, making it a popular destination for both locals and tourists.\n
Elarian Freiseur - A short stroll from Sapore di Mare is Elarian Freiseur, a boutique hair salon known for its chic style and innovative hair treatments. This salon stands out with its modern design, featuring sleek chairs, ambient lighting, and an array of plants that add a touch of greenery and freshness.
The team at Elarian Freiseur is comprised of highly skilled stylists and colorists who are experts in the latest hair trends. They offer personalized consultations to each client, ensuring a customized experience that meets individual style preferences. From classic cuts to avant-garde hair coloring, the salon is a hub for those seeking a transformative hair experience.
Elarian Freiseur also places a high emphasis on using eco-friendly and sustainable hair products, aligning with the city's growing environmental consciousness. This commitment to quality and sustainability has earned it a loyal clientele who appreciate the salon's dedication to both style and the environment.
        
        ";

        let chunker = CumulativeChunker::default();
        chunker._chunk(text);
    }
}
