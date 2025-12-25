use std::path::{Path, PathBuf};
use std::{collections::HashMap, fs::File};

use fancy_regex::Regex as FancyRegex;
use memmap2::MmapOptions;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use regex::Regex;
use thread_local::ThreadLocal;

use crate::utils::create_progress_bar;

fn get_pre_tokenizer_pattern() -> &'static FancyRegex {
    static TLS: ThreadLocal<FancyRegex> = ThreadLocal::new();
    TLS.get_or(|| {
        let pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        // 这里 panic 是安全的，因为 pattern 是硬编码的，测试过没问题就行
        FancyRegex::new(pattern).expect("Invalid Regex Pattern")
    })
}

fn get_special_tokens_pattern(special_tokens: &[String]) -> anyhow::Result<Regex, anyhow::Error> {
    let escaped_tokens: Vec<String> = special_tokens.iter().map(|t| regex::escape(t)).collect();

    let pattern = escaped_tokens.join("|");

    let re = Regex::new(&pattern)?;

    Ok(re)
}

fn worker(
    chunk: &str,
    special_tokens_re: &Regex,
) -> anyhow::Result<HashMap<Vec<u8>, u32>, anyhow::Error> {
    let mut token_count: HashMap<Vec<u8>, u32> = HashMap::new();

    let sub_chunks = special_tokens_re.split(chunk).filter(|s| !s.is_empty());

    for sub_chunk in sub_chunks {
        for m in get_pre_tokenizer_pattern().find_iter(&sub_chunk) {
            let bytes = m?.as_str().as_bytes();
            if let Some(c) = token_count.get_mut(bytes) {
                *c += 1;
            } else {
                token_count.insert(bytes.to_vec(), 1);
            }
        }
    }

    Ok(token_count)
}

fn merge_token_count(
    a: anyhow::Result<HashMap<Vec<u8>, u32>>,
    b: anyhow::Result<HashMap<Vec<u8>, u32>>,
) -> anyhow::Result<HashMap<Vec<u8>, u32>> {
    let (mut map_a, map_b) = (a?, b?);
    for (k, v) in map_b {
        *map_a.entry(k).or_default() += v;
    }
    Ok(map_a)
}

pub fn pre_tokenize_impl(
    py: Python,
    path: &Path,
    special_tokens: &[String],
    boundaries: &[(usize, usize)],
    num_threads: usize,
) -> PyResult<HashMap<Vec<u8>, u32>> {
    let f = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&f)? };

    let special_tokens_re = get_special_tokens_pattern(special_tokens)
        .map_err(|e| PyRuntimeError::new_err(format!("Rust Error: {:?}", e)))?;

    // 创建进度条
    let pb = create_progress_bar(num_threads as u64, "Pre-tokenizing");

    // 构建局部线程池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.min(64))
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // 在这个池子里运行代码
    // pool.install 会“劫持”闭包里的所有 rayon 操作(如 par_iter)使用这个池子
    let token_count = py.detach(|| {
        pool.install(|| {
            boundaries
                .par_iter()
                .map(|&(start, end)| {
                    // 每个线程处理一个切片
                    let slice = &mmap[start..end];

                    let chunk = std::str::from_utf8(slice).map_err(|e| {
                        anyhow::anyhow!(format!("UTF-8 error at {}-{}: {}", start, end, e))
                    })?;

                    let chunk_token_count = worker(chunk, &special_tokens_re);

                    pb.inc(1);

                    chunk_token_count
                })
                .reduce(|| Ok(HashMap::new()), merge_token_count)
        })
    });

    pb.finish();

    token_count.map_err(|e| PyRuntimeError::new_err(format!("Rust error: {:?}", e)))
}

#[pyfunction]
pub fn pre_tokenize(
    py: Python,
    path: PathBuf,
    special_tokens: Vec<String>,
    boundaries: Vec<(usize, usize)>,
    num_threads: usize,
) -> PyResult<HashMap<Vec<u8>, u32>> {
    Ok(pre_tokenize_impl(
        py,
        &path,
        &special_tokens,
        &boundaries,
        num_threads,
    )?)
}
