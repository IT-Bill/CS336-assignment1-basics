use std::collections::HashMap;

use pyo3::prelude::*;
use std::collections::BinaryHeap;
use indicatif::{ProgressIterator, ProgressBar, ProgressStyle};

struct Symbol {
    id: usize,
    prev: Option<usize>,
    next: Option<usize>,
    is_alive: bool,
}

struct Token {
    symbols: Vec<Symbol>,
    count: u32,
}

// 如果函数只有一个输入引用参数，那么所有输出引用的生命周期自动和它绑定
fn id_pair_to_bytes_pair(symbol_bytes: &Vec<Vec<u8>>, id_pair: (usize, usize)) -> (Vec<u8>, Vec<u8>) {
    (symbol_bytes[id_pair.0].clone(), symbol_bytes[id_pair.1].clone())
}



#[pyfunction]
pub fn merge(token_count: HashMap<Vec<u8>, u32>, num_merges: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    // symbol id -> bytes
    let mut symbol_bytes = (0..256).map(|i| vec![i as u8]).collect::<Vec<_>>();

    let mut merged_pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

    // (A, B) -> (token_idx, sym_idx)
    let mut pair_occurrences: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();

    // (A, B) -> count
    let mut pair_count: HashMap<(usize, usize), u32> = HashMap::new();

    // (token_idx, sym_idx) -> sym
    let mut token_list: Vec<Token> = Vec::new();

    let pb = ProgressBar::new(num_merges as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    for (token_idx, (sym_ids, count)) in token_count.iter().enumerate() {
        let mut symbols: Vec<Symbol> = Vec::new();
        let length = sym_ids.len();

        for (sym_idx, sym_id) in sym_ids.iter().enumerate() {

            let mut symbol = Symbol {
                id: *sym_id as usize,
                prev: None,
                next: None,
                is_alive: true,
            };
            if sym_idx + 1 < length {
                let pair = (*sym_id as usize, sym_ids[sym_idx + 1] as usize);
                pair_occurrences
                    .entry(pair)
                    .or_default()
                    .push((token_idx, sym_idx));
                *pair_count.entry(pair).or_default() += count;
            }

            // !不能写 sym_idx - 1 >= 0，越界
            if sym_idx >= 1 as usize {
                symbol.prev = Some(sym_idx - 1);
                symbols[sym_idx - 1].next = Some(sym_idx);
            }

            symbols.push(symbol);
        }

        token_list.push(Token {
            symbols,
            count: *count,
        })
    }

    let mut pair_count_heap = BinaryHeap::from(
        pair_count
            .iter()
            .map(|(&pair, &count)| (count, id_pair_to_bytes_pair(&symbol_bytes, pair), pair))
            .collect::<Vec<_>>(),
    );

    for _ in (0..num_merges).progress_with(pb) {
        let mut most_common_pair: Option<(usize, usize)> = None;

        while let Some((count,_ , pair)) = pair_count_heap.pop() {
            if let Some(&actual_count) = pair_count.get(&pair) {
                if count == actual_count {
                    most_common_pair = Some(pair);
                    break;
                }
            }
        }

        
        let Some((sym_a_id, sym_b_id)) = most_common_pair else {
            break;
        };

        let merged_symbol_id = symbol_bytes.len();

        // --- 步骤 1: 先生成新数据 (只读取，不修改) ---
        // 使用 slice concat 生成一个新的 Vec<u8>，这会自动克隆数据
        // 这一行执行完后，对 symbol_bytes 的借用就立即结束了
        let new_symbol_bytes = [
            &symbol_bytes[sym_a_id][..],
            &symbol_bytes[sym_b_id][..],
        ].concat();

        // 顺便把要存入 merged_pairs 的数据也准备好 (克隆出来)
        let new_merged_pair = (
            symbol_bytes[sym_a_id].clone(),
            symbol_bytes[sym_b_id].clone()
        );

        // --- 步骤 2: 现在不再持有 symbol_bytes 的引用了，可以安全修改 ---
        symbol_bytes.push(new_symbol_bytes);
        merged_pairs.push(new_merged_pair);

        // let occurrences = &pair_occurrences[&(sym_a_id, sym_b_id)].clone();
        let Some(occurrences) = pair_occurrences.get(&(sym_a_id, sym_b_id)).cloned() else {
            continue;
        };
        let mut valid_occ_indices: Vec<usize> = Vec::new();

        for (occ_idx, (token_idx, sym_idx)) in occurrences.iter().enumerate() {
            let token = &mut token_list[*token_idx];
            let a = &token.symbols[*sym_idx];

            if !a.is_alive {
                continue;
            }

            let Some(b_idx) = a.next else {
                continue;
            };

            let b = &token.symbols[b_idx];

            if !b.is_alive || b.id != sym_b_id {
                continue;
            }

            valid_occ_indices.push(occ_idx);

            let l_idx = a.prev;
            let r_idx = b.next;

            // Update symbol A to symbol AB
            token.symbols[*sym_idx].id = merged_symbol_id;
            // Skip b, remember this is double-link list
            token.symbols[*sym_idx].next = r_idx;
            if let Some(r_idx) = r_idx {
                token.symbols[r_idx].prev = Some(*sym_idx);
            }

            token.symbols[b_idx].is_alive = false;
            
            
            if let Some(l_idx) = l_idx {
                // Decrease (L, A)
                let la_id = (token.symbols[l_idx].id, sym_a_id);
                *pair_count.entry(la_id).or_default() -= token.count;
                pair_count_heap.push((pair_count[&la_id], id_pair_to_bytes_pair(&symbol_bytes, la_id), la_id));

                // Increase (L, AB)
                let lab_id = (token.symbols[l_idx].id, merged_symbol_id);
                *pair_count.entry(lab_id).or_default() += token.count;
                pair_occurrences.entry(lab_id).or_default().push((*token_idx, l_idx));
                pair_count_heap.push((pair_count[&lab_id], id_pair_to_bytes_pair(&symbol_bytes, lab_id), lab_id));
            }

            if let Some(r_idx) = r_idx {
                // Decrease (B, R)
                let br_id = (token.symbols[b_idx].id, token.symbols[r_idx].id);
                *pair_count.entry(br_id).or_default() -= token.count;
                pair_count_heap.push((pair_count[&br_id], id_pair_to_bytes_pair(&symbol_bytes, br_id), br_id));

                // Increase (AB, R)
                let abr_id = (merged_symbol_id, token.symbols[r_idx].id);
                *pair_count.entry(abr_id).or_default() += token.count;
                pair_occurrences.entry(abr_id).or_default().push((*token_idx, *sym_idx));
                pair_count_heap.push((pair_count[&abr_id], id_pair_to_bytes_pair(&symbol_bytes, abr_id), abr_id));
            }
            
        }

        let kept_occurrences: Vec<(usize, usize)> = valid_occ_indices
            .iter()
            .map(|&i| occurrences[i])
            .collect();

        pair_occurrences.insert((sym_a_id, sym_b_id), kept_occurrences);

        // Remove (A, B) from pair_count
        pair_count.remove(&(sym_a_id, sym_b_id));

    }

    merged_pairs
}
