use std::{collections::HashMap, fs::File, io, path::PathBuf};

use memmap2::MmapOptions;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use regex::Regex;
use fancy_regex::Regex as FancyRegex;

fn remove_special_tokens(chunk: &str, special_tokens: &Vec<String>) -> Result<Vec<String>, anyhow::Error> {
    let escaped_tokens: Vec<String> = special_tokens
        .iter()
        .map(|t| regex::escape(t))
        .collect();
    
    let pattern = escaped_tokens.join("|");

    if pattern.is_empty() {
        return Ok(vec![chunk.to_string()]);
    }

    let re = Regex::new(&pattern)?;

    let result: Vec<String> = re
        .split(chunk)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    Ok(result)

}

fn worker(chunk: &str, special_tokens: &Vec<String>) -> Result<HashMap<Vec<u8>, u32>, anyhow::Error> {
    let pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    
    // TODO: cache
    let re = FancyRegex::new(pattern)?;

    let mut token_count: HashMap<Vec<u8>, u32> = HashMap::new();

    let sub_chunks = remove_special_tokens(chunk, special_tokens)?;

    for sub_chunk in sub_chunks {
        for m in re.find_iter(&sub_chunk) {
            *token_count.entry(m?.as_str().as_bytes().to_vec()).or_default() += 1;
        }
    }

    Ok(token_count)

}

#[pyfunction]
pub fn pre_tokenize(
    path: PathBuf,
    special_tokens: Vec<String>,
    boundaries: Vec<(usize, usize)>,
) -> PyResult<HashMap<Vec<u8>, u32>> {
    let f = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&f)? };

    let mut token_count: HashMap<Vec<u8>, u32> = HashMap::new();

    for (start, end) in boundaries {
        let slice = &mmap[start..end];
        // 把 &[u8] 解释成 UTF-8 文本，并将错误类型转换
        let chunk = std::str::from_utf8(slice)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        let chunk_token_count = worker(chunk, &special_tokens)
            .map_err(|e| PyRuntimeError::new_err(format!("Rust worker failed: {}", e)))?;

        for (token, chunk) in chunk_token_count {
            *token_count.entry(token).or_default() += chunk;
        }

    }

    Ok(token_count)
}
