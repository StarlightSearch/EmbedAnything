use strum::EnumString;

use super::pooling::Pooling;

use super::model_info::ModelInfo;

// use super::quantization::QuantizationMode;

use std::{collections::HashMap, fmt::Display, sync::OnceLock};

/// Lazy static list of all available models.
static MODEL_MAP: OnceLock<HashMap<ONNXModel, ModelInfo<ONNXModel>>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, Hash, EnumString)]
pub enum ONNXModel {
    /// sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2,
    /// Quantized sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2Q,
    /// sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2,
    /// Quantized sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2Q,
    /// BAAI/bge-base-en-v1.5
    BGEBaseENV15,
    /// Quantized BAAI/bge-base-en-v1.5
    BGEBaseENV15Q,
    /// BAAI/bge-large-en-v1.5
    BGELargeENV15,
    /// Quantized BAAI/bge-large-en-v1.5
    BGELargeENV15Q,
    /// BAAI/bge-small-en-v1.5 - Default
    BGESmallENV15,
    /// Quantized BAAI/bge-small-en-v1.5
    BGESmallENV15Q,
    /// nomic-ai/nomic-embed-text-v1
    NomicEmbedTextV1,
    /// nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15,
    /// Quantized v1.5 nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15Q,
    /// sentence-transformers/paraphrase-MiniLM-L6-v2
    ParaphraseMLMiniLML12V2,
    /// Quantized sentence-transformers/paraphrase-MiniLM-L6-v2
    ParaphraseMLMiniLML12V2Q,
    /// sentence-transformers/paraphrase-mpnet-base-v2
    ParaphraseMLMpnetBaseV2,
    /// BAAI/bge-small-zh-v1.5
    BGESmallZHV15,
    /// intfloat/multilingual-e5-small
    MultilingualE5Small,
    /// intfloat/multilingual-e5-base
    MultilingualE5Base,
    /// intfloat/multilingual-e5-large
    MultilingualE5Large,
    /// mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1,
    /// Quantized mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1Q,
    /// Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15,
    /// Quantized Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15Q,
    /// Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15,
    /// Quantized Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15Q,

    /// jinaai/jina-embeddings-v2-small-en
    JINAV2SMALLEN,
    /// jinaai/jina-embeddings-v2-base-en
    JINAV2BASEEN,
    /// jinaai/jina-embeddings-v2-large-en
    JINAV2LARGEEN,
    

}

// impl From<&str> for ONNXModel {
//     fn from(s: &str) -> Self {
//         ONNXModel::from(s)
//     }
// }

/// Centralized function to initialize the models map.
fn init_models_map() -> HashMap<ONNXModel, ModelInfo<ONNXModel>> {
    let models_list = vec![
        ModelInfo {
            model: ONNXModel::AllMiniLML6V2,
            dim: 384,
            description: String::from("Sentence Transformer model, MiniLM-L6-v2"),
            hf_model_id: String::from("sentence-transformers/all-MiniLM-L6-v2"),
            model_code: String::from("Qdrant/all-MiniLM-L6-v2-onnx"),
            model_file: String::from("model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::AllMiniLML6V2Q,
            dim: 384,
            description: String::from("Quantized Sentence Transformer model, MiniLM-L6-v2"),
            hf_model_id: String::from("Xenova/all-MiniLM-L6-v2"),
            model_code: String::from("Xenova/all-MiniLM-L6-v2"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
        ModelInfo {
            model: ONNXModel::AllMiniLML12V2,
            dim: 384,
            description: String::from("Sentence Transformer model, MiniLM-L12-v2"),
            hf_model_id: String::from("sentence-transformers/all-MiniLM-L12-v2"),
            model_code: String::from("Xenova/all-MiniLM-L12-v2"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::AllMiniLML12V2Q,
            dim: 384,
            description: String::from("Quantized Sentence Transformer model, MiniLM-L12-v2"),
            hf_model_id: String::from("Xenova/all-MiniLM-L12-v2"),
            model_code: String::from("Xenova/all-MiniLM-L12-v2"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
        ModelInfo {
            model: ONNXModel::BGEBaseENV15,
            dim: 768,
            description: String::from("v1.5 release of the base English model"),
            hf_model_id: String::from("BAAI/bge-base-en-v1.5"),
            model_code: String::from("Xenova/bge-base-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::BGEBaseENV15Q,
            dim: 768,
            description: String::from("Quantized v1.5 release of the large English model"),
            hf_model_id: String::from("Qdrant/bge-base-en-v1.5-onnx-Q"),
            model_code: String::from("Qdrant/bge-base-en-v1.5-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
        },
        ModelInfo {
            model: ONNXModel::BGELargeENV15,
            dim: 1024,
            description: String::from("v1.5 release of the large English model"),
            hf_model_id: String::from("BAAI/bge-large-en-v1.5"),
            model_code: String::from("Xenova/bge-large-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::BGELargeENV15Q,
            dim: 1024,
            description: String::from("Quantized v1.5 release of the large English model"),
            hf_model_id: String::from("Qdrant/bge-large-en-v1.5-onnx-Q"),
            model_code: String::from("Qdrant/bge-large-en-v1.5-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
        },
        ModelInfo {
            model: ONNXModel::BGESmallENV15,
            dim: 384,
            description: String::from("v1.5 release of the fast and default English model"),
            hf_model_id: String::from("BAAI/bge-small-en-v1.5"),
            model_code: String::from("Xenova/bge-small-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::BGESmallENV15Q,
            dim: 384,
            description: String::from(
                "Quantized v1.5 release of the fast and default English model",
            ),
            hf_model_id: String::from("Qdrant/bge-small-en-v1.5-onnx-Q"),
            model_code: String::from("Qdrant/bge-small-en-v1.5-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
        },
        ModelInfo {
            model: ONNXModel::NomicEmbedTextV1,
            dim: 768,
            description: String::from("8192 context length english model"),
            hf_model_id: String::from("nomic-ai/nomic-embed-text-v1"),
            model_code: String::from("nomic-ai/nomic-embed-text-v1"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::NomicEmbedTextV15,
            dim: 768,
            description: String::from("v1.5 release of the 8192 context length english model"),
            hf_model_id: String::from("nomic-ai/nomic-embed-text-v1.5"),
            model_code: String::from("nomic-ai/nomic-embed-text-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::NomicEmbedTextV15Q,
            dim: 768,
            description: String::from(
                "Quantized v1.5 release of the 8192 context length english model",
            ),
            hf_model_id: String::from("Qdrant/nomic-embed-text-v1.5-onnx-Q"),
            model_code: String::from("Qdrant/nomic-embed-text-v1.5-onnx-Q"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
        ModelInfo {
            model: ONNXModel::ParaphraseMLMiniLML12V2Q,
            dim: 384,
            description: String::from("Quantized Multi-lingual model"),
            hf_model_id: String::from("Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"),
            model_code: String::from("Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
        },
        ModelInfo {
            model: ONNXModel::ParaphraseMLMiniLML12V2,
            dim: 384,
            description: String::from("Multi-lingual model"),
            hf_model_id: String::from("sentence-transformers/paraphrase-MiniLM-L6-v2"),
            model_code: String::from("Xenova/paraphrase-multilingual-MiniLM-L12-v2"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::ParaphraseMLMpnetBaseV2,
            dim: 768,
            description: String::from(
                "Sentence-transformers model for tasks like clustering or semantic search",
            ),
            hf_model_id: String::from(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            ),
            model_code: String::from("Xenova/paraphrase-multilingual-mpnet-base-v2"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::BGESmallZHV15,
            dim: 512,
            description: String::from("v1.5 release of the small Chinese model"),
            hf_model_id: String::from("BAAI/bge-small-zh-v1.5"),
            model_code: String::from("Xenova/bge-small-zh-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::MultilingualE5Small,
            dim: 384,
            description: String::from("Small model of multilingual E5 Text Embeddings"),
            hf_model_id: String::from("intfloat/multilingual-e5-small"),
            model_code: String::from("intfloat/multilingual-e5-small"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::MultilingualE5Base,
            dim: 768,
            description: String::from("Base model of multilingual E5 Text Embeddings"),
            hf_model_id: String::from("intfloat/multilingual-e5-base"),
            model_code: String::from("intfloat/multilingual-e5-base"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::MultilingualE5Large,
            dim: 1024,
            description: String::from("Large model of multilingual E5 Text Embeddings"),
            hf_model_id: String::from("intfloat/multilingual-e5-large"),
            model_code: String::from("Qdrant/multilingual-e5-large-onnx"),
            model_file: String::from("model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::MxbaiEmbedLargeV1,
            dim: 1024,
            description: String::from("Large English embedding model from MixedBreed.ai"),
            hf_model_id: String::from("mixedbread-ai/mxbai-embed-large-v1"),
            model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::MxbaiEmbedLargeV1Q,
            dim: 1024,
            description: String::from("Quantized Large English embedding model from MixedBreed.ai"),
            hf_model_id: String::from("mixedbread-ai/mxbai-embed-large-v1"),
            model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
        ModelInfo {
            model: ONNXModel::GTEBaseENV15,
            dim: 768,
            description: String::from("Large multilingual embedding model from Alibaba"),
            hf_model_id: String::from("Alibaba-NLP/gte-base-en-v1.5"),
            model_code: String::from("Alibaba-NLP/gte-base-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::GTEBaseENV15Q,
            dim: 768,
            description: String::from("Quantized Large multilingual embedding model from Alibaba"),
            hf_model_id: String::from("Alibaba-NLP/gte-base-en-v1.5"),
            model_code: String::from("Alibaba-NLP/gte-base-en-v1.5"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
        ModelInfo {
            model: ONNXModel::GTELargeENV15,
            dim: 1024,
            description: String::from("Large multilingual embedding model from Alibaba"),
            hf_model_id: String::from("Alibaba-NLP/gte-large-en-v1.5"),
            model_code: String::from("Alibaba-NLP/gte-large-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: ONNXModel::GTELargeENV15Q,
            dim: 1024,
            description: String::from("Quantized Large multilingual embedding model from Alibaba"),
            hf_model_id: String::from("Alibaba-NLP/gte-large-en-v1.5"),
            model_code: String::from("Alibaba-NLP/gte-large-en-v1.5"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
    ];

    // TODO: Use when out in stable
    // assert_eq!(
    //     std::mem::variant_count::<ONNXModel>(),
    //     models_list.len(),
    //     "models::models() is not exhaustive"
    // );

    models_list
        .into_iter()
        .fold(HashMap::new(), |mut map, model| {
            // Insert the model into the map
            map.insert(model.model.clone(), model);
            map
        })
}

/// Get a map of all available models.
pub fn models_map() -> &'static HashMap<ONNXModel, ModelInfo<ONNXModel>> {
    MODEL_MAP.get_or_init(init_models_map)
}

/// Get model information by model code.
pub fn get_model_info(model: &ONNXModel) -> Option<&ModelInfo<ONNXModel>> {
    models_map().get(model)
}

pub fn get_model_info_by_hf_id(hf_model_id: &str) -> Option<&ModelInfo<ONNXModel>> {
    models_map()
        .values()
        .find(|model| model.hf_model_id == hf_model_id)
}


/// Get a list of all available models.
///
/// This will assign new memory to the models list; where possible, use
/// [`models_map`] instead.
pub fn models_list() -> Vec<ModelInfo<ONNXModel>> {
    models_map().values().cloned().collect()
}

impl ONNXModel {
    pub fn get_default_pooling_method(&self) -> Option<Pooling> {
        match self {
            ONNXModel::AllMiniLML6V2 => Some(Pooling::Mean),
            ONNXModel::AllMiniLML6V2Q => Some(Pooling::Mean),
            ONNXModel::AllMiniLML12V2 => Some(Pooling::Mean),
            ONNXModel::AllMiniLML12V2Q => Some(Pooling::Mean),

            ONNXModel::BGEBaseENV15 => Some(Pooling::Cls),
            ONNXModel::BGEBaseENV15Q => Some(Pooling::Cls),
            ONNXModel::BGELargeENV15 => Some(Pooling::Cls),
            ONNXModel::BGELargeENV15Q => Some(Pooling::Cls),
            ONNXModel::BGESmallENV15 => Some(Pooling::Cls),
            ONNXModel::BGESmallENV15Q => Some(Pooling::Cls),
            ONNXModel::BGESmallZHV15 => Some(Pooling::Cls),

            ONNXModel::NomicEmbedTextV1 => Some(Pooling::Mean),
            ONNXModel::NomicEmbedTextV15 => Some(Pooling::Mean),
            ONNXModel::NomicEmbedTextV15Q => Some(Pooling::Mean),

            ONNXModel::ParaphraseMLMiniLML12V2 => Some(Pooling::Mean),
            ONNXModel::ParaphraseMLMiniLML12V2Q => Some(Pooling::Mean),
            ONNXModel::ParaphraseMLMpnetBaseV2 => Some(Pooling::Mean),

            ONNXModel::MultilingualE5Base => Some(Pooling::Mean),
            ONNXModel::MultilingualE5Small => Some(Pooling::Mean),
            ONNXModel::MultilingualE5Large => Some(Pooling::Mean),

            ONNXModel::MxbaiEmbedLargeV1 => Some(Pooling::Cls),
            ONNXModel::MxbaiEmbedLargeV1Q => Some(Pooling::Cls),

            ONNXModel::GTEBaseENV15 => Some(Pooling::Cls),
            ONNXModel::GTEBaseENV15Q => Some(Pooling::Cls),
            ONNXModel::GTELargeENV15 => Some(Pooling::Cls),
            ONNXModel::GTELargeENV15Q => Some(Pooling::Cls),

            ONNXModel::JINAV2SMALLEN => Some(Pooling::Mean),
            ONNXModel::JINAV2BASEEN => Some(Pooling::Mean),
            ONNXModel::JINAV2LARGEEN => Some(Pooling::Mean),
        }
    }

    /// Get the quantization mode of the model.
    ///
    /// Any models with a `Q` suffix in their name are quantized models.
    ///
    /// Currently only 6 supported models have dynamic quantization:
    /// - Alibaba-NLP/gte-base-en-v1.5
    /// - Alibaba-NLP/gte-large-en-v1.5
    /// - mixedbread-ai/mxbai-embed-large-v1
    /// - nomic-ai/nomic-embed-text-v1.5
    /// - Xenova/all-MiniLM-L12-v2
    /// - Xenova/all-MiniLM-L6-v2
    ///
    // TODO: Update this list when more models are added
    pub fn get_quantization_mode(&self) -> Result<(), anyhow::Error> {
        // match self {
        //     ONNXModel::AllMiniLML6V2Q => QuantizationMode::Dynamic,
        //     ONNXModel::AllMiniLML12V2Q => QuantizationMode::Dynamic,
        //     ONNXModel::BGEBaseENV15Q => QuantizationMode::Static,
        //     ONNXModel::BGELargeENV15Q => QuantizationMode::Static,
        //     ONNXModel::BGESmallENV15Q => QuantizationMode::Static,
        //     ONNXModel::NomicEmbedTextV15Q => QuantizationMode::Dynamic,
        //     ONNXModel::ParaphraseMLMiniLML12V2Q => QuantizationMode::Static,
        //     ONNXModel::MxbaiEmbedLargeV1Q => QuantizationMode::Dynamic,
        //     ONNXModel::GTEBaseENV15Q => QuantizationMode::Dynamic,
        //     ONNXModel::GTELargeENV15Q => QuantizationMode::Dynamic,
        //     _ => QuantizationMode::None,
        // }
        unimplemented!()
    }
}

impl Display for ONNXModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = get_model_info(self).expect("Model not found.");
        write!(f, "{}", model_info.model_code)
    }
}
