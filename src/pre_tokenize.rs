use std::{collections::HashMap, fs::File, path::PathBuf, time::Instant};

use fancy_regex::Regex as FancyRegex;
use memmap2::MmapOptions;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use regex::Regex;
use indicatif::{ProgressBar, ProgressStyle};

fn remove_special_tokens(
    chunk: &str,
    special_tokens: &Vec<String>,
) -> anyhow::Result<Vec<String>, anyhow::Error> {
    let escaped_tokens: Vec<String> = special_tokens.iter().map(|t| regex::escape(t)).collect();

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

fn worker(
    chunk: &str,
    special_tokens: &Vec<String>,
) -> anyhow::Result<HashMap<Vec<u8>, u32>, anyhow::Error> {
    let pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    // TODO: cache
    let re = FancyRegex::new(pattern)?;

    let mut token_count: HashMap<Vec<u8>, u32> = HashMap::new();

    let sub_chunks = remove_special_tokens(chunk, special_tokens)?;

    for sub_chunk in sub_chunks {
        for m in re.find_iter(&sub_chunk) {
            *token_count
                .entry(m?.as_str().as_bytes().to_vec())
                .or_default() += 1;
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

#[pyfunction]
pub fn pre_tokenize(
    py: Python,
    path: PathBuf,
    special_tokens: Vec<String>,
    boundaries: Vec<(usize, usize)>,
    num_threads: usize,
) -> PyResult<HashMap<Vec<u8>, u32>> {
    let start_time = Instant::now();

    let f = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&f)? };

    // 1. 【新增】手动创建进度条
    let pb = ProgressBar::new(boundaries.len() as u64);
    pb.set_style(ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})"
    ).unwrap().progress_chars("#>-"));

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

                    let chunk_token_count = worker(chunk, &special_tokens);

                    pb.inc(1);

                    chunk_token_count
                })
                .reduce(|| Ok(HashMap::new()), merge_token_count)
        })
    });

    let duration = start_time.elapsed();
    println!("🦀 Rust 纯计算耗时: {:.4} 秒", duration.as_secs_f64());

    token_count.map_err(|e| PyRuntimeError::new_err(format!("Rust error: {:?}", e)))
}
