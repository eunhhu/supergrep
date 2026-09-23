//! The reusable engine behind the `supergrep` command-line interface.
//!
//! The public surface deliberately keeps model scoring separate from discovery
//! and ranking so deterministic tests can use a fake scorer without making the
//! product path anything other than local ONNX inference.

pub mod chunk;
pub mod cli;
pub mod discovery;
pub mod error;
pub mod fitting;
pub mod lexical;
pub mod model;
pub mod output;
pub mod search;
pub mod source;

pub use error::{Result, SupergrepError};
