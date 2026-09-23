use std::path::Path;

use tokenizers::Tokenizer;

use crate::{Result, SupergrepError};

/// Immutable input rules that are pinned with each model profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerContract {
    pub max_pair_tokens: usize,
    pub max_query_tokens: usize,
    pub pad_id: u32,
}

impl Default for TokenizerContract {
    fn default() -> Self {
        Self {
            max_pair_tokens: 256,
            max_query_tokens: 64,
            pad_id: 0,
        }
    }
}

/// A padded, batch-major set of model inputs.  All vectors have
/// `batch_len * sequence_len` entries and use signed 64-bit IDs as required by
/// the BERT/XLM-R ONNX graphs used by v0.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairBatch {
    pub batch_len: usize,
    pub sequence_len: usize,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
}

#[derive(Clone)]
pub struct PairTokenizer {
    inner: Tokenizer,
    contract: TokenizerContract,
}

/// Reuses the model-tokenized query while checking many source passages.
/// The profile tokenizer still supplies every pair's exact special tokens.
pub struct PreparedPairQuery<'a> {
    tokenizer: &'a PairTokenizer,
    query_encoding: tokenizers::Encoding,
}

impl PreparedPairQuery<'_> {
    pub fn token_count(&self, passage: &str) -> Result<usize> {
        let passage_encoding = self
            .tokenizer
            .inner
            .encode(passage, false)
            .map_err(|error| {
                SupergrepError::model(format!("could not tokenize passage: {error}"))
            })?;
        let pair = self
            .tokenizer
            .inner
            .post_process(self.query_encoding.clone(), Some(passage_encoding), true)
            .map_err(|error| {
                SupergrepError::model(format!("could not assemble query/passage pair: {error}"))
            })?;
        Ok(pair.get_ids().len())
    }
}

impl PairTokenizer {
    pub fn from_file(path: &Path, contract: TokenizerContract) -> Result<Self> {
        SupergrepError::require_file(path)?;
        let mut inner = Tokenizer::from_file(path).map_err(|error| {
            SupergrepError::model(format!(
                "could not load tokenizer {}: {error}",
                path.display()
            ))
        })?;

        // A serialized Hugging Face tokenizer can carry its producer's
        // fixed-length padding and truncation settings (the GTE export, for
        // example, has both set to 512).  Those are not our inference
        // contract: we must reject overlong pairs before ONNX rather than
        // silently truncate or mistake producer padding for query length.
        inner.with_truncation(None).map_err(|error| {
            SupergrepError::model(format!("could not disable tokenizer truncation: {error}"))
        })?;
        inner.with_padding(None);
        Ok(Self { inner, contract })
    }

    pub fn contract(&self) -> TokenizerContract {
        self.contract
    }

    /// Validates a query against the model-specific, special-token-inclusive
    /// limit without truncating it.  Search calls this once before it starts
    /// splitting source passages, so an invalid query can never turn into a
    /// partially searched result set.
    pub fn validate_query(&self, query: &str) -> Result<()> {
        let query_encoding = self.encode_query(query)?;
        if query_encoding.get_ids().len() > self.contract.max_query_tokens {
            return Err(SupergrepError::Input(format!(
                "query encodes to {} tokens including special tokens; maximum is {}",
                query_encoding.get_ids().len(),
                self.contract.max_query_tokens
            )));
        }
        Ok(())
    }

    /// Returns the exact model pair length including the tokenizer's special
    /// tokens.  It never truncates.  This is deliberately available to the
    /// coordinate-preserving fitting layer, which must make every chunk fit
    /// before candidate selection rather than let ONNX reject a late batch.
    pub fn pair_token_count(&self, query: &str, passage: &str) -> Result<usize> {
        self.validate_query(query)?;
        Ok(self.encode_pair(query, passage)?.get_ids().len())
    }

    /// Validates and tokenizes the query once for repeated exact pair fitting.
    /// The returned counter borrows this tokenizer and cannot outlive it.
    pub fn prepare_pair_query(&self, query: &str) -> Result<PreparedPairQuery<'_>> {
        self.validate_query(query)?;
        let query_encoding = self
            .inner
            .encode(query, false)
            .map_err(|error| SupergrepError::model(format!("could not tokenize query: {error}")))?;
        Ok(PreparedPairQuery {
            tokenizer: self,
            query_encoding,
        })
    }

    /// Encodes a query/passage batch with model special tokens.  It never
    /// truncates: a caller must split a source chunk before scoring it.
    pub fn encode_batch(&self, query: &str, passages: &[String]) -> Result<PairBatch> {
        if passages.is_empty() {
            return Ok(PairBatch {
                batch_len: 0,
                sequence_len: 0,
                input_ids: Vec::new(),
                attention_mask: Vec::new(),
                token_type_ids: Vec::new(),
            });
        }

        self.validate_query(query)?;

        let mut rows = Vec::with_capacity(passages.len());
        let mut sequence_len = 0usize;
        for passage in passages {
            let encoding = self.encode_pair(query, passage)?;
            let length = encoding.get_ids().len();
            if length > self.contract.max_pair_tokens {
                return Err(SupergrepError::Input(format!(
                    "query/passage pair encodes to {length} tokens including special tokens; maximum is {}",
                    self.contract.max_pair_tokens
                )));
            }
            sequence_len = sequence_len.max(length);
            rows.push((
                encoding.get_ids().to_vec(),
                encoding.get_attention_mask().to_vec(),
                encoding.get_type_ids().to_vec(),
            ));
        }

        let mut input_ids = Vec::with_capacity(rows.len() * sequence_len);
        let mut attention_mask = Vec::with_capacity(rows.len() * sequence_len);
        let mut token_type_ids = Vec::with_capacity(rows.len() * sequence_len);
        for (ids, masks, types) in rows {
            input_ids.extend(ids.iter().map(|&id| i64::from(id)));
            attention_mask.extend(masks.iter().map(|&mask| i64::from(mask)));
            token_type_ids.extend(types.iter().map(|&kind| i64::from(kind)));
            let padding = sequence_len - ids.len();
            input_ids.extend(std::iter::repeat(i64::from(self.contract.pad_id)).take(padding));
            attention_mask.extend(std::iter::repeat(0).take(padding));
            token_type_ids.extend(std::iter::repeat(0).take(padding));
        }

        Ok(PairBatch {
            batch_len: passages.len(),
            sequence_len,
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }

    fn encode_query(&self, query: &str) -> Result<tokenizers::Encoding> {
        self.inner
            .encode(query, true)
            .map_err(|error| SupergrepError::model(format!("could not tokenize query: {error}")))
    }

    fn encode_pair(&self, query: &str, passage: &str) -> Result<tokenizers::Encoding> {
        self.inner.encode((query, passage), true).map_err(|error| {
            SupergrepError::model(format!("could not tokenize query/passage pair: {error}"))
        })
    }
}
