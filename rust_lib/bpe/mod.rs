use pyo3::prelude::*;
use std::path::PathBuf;

pub mod merge;
pub mod pre_tokenize;

#[pyfunction]
pub fn train(
    py: Python,
    path: PathBuf,
    special_tokens: Vec<String>,
    boundaries: Vec<(usize, usize)>,
    num_merges: usize,
    num_threads: usize,
) -> PyResult<Vec<(Vec<u8>, Vec<u8>)>> {
    let token_count = self::pre_tokenize::pre_tokenize_impl(
        py,
        &path,
        &special_tokens,
        &boundaries,
        num_threads,
    )?;

    let merged_pairs = self::merge::merge_impl(&token_count, num_merges)?;

    Ok(merged_pairs)
}
