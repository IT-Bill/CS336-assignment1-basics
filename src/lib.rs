use pyo3::prelude::*;

mod bpe;
mod utils;

/// A Python module implemented in Rust.
#[pymodule]
fn rust_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 1. 创建名为 "bpe" 的子模块
    let bpe_module = PyModule::new(m.py(), "bpe")?;

    // 2. 将函数添加到子模块 (注意 wrap_pyfunction! 的第二个参数变成了 &bpe_module)
    bpe_module.add_function(wrap_pyfunction!(bpe::pre_tokenize::pre_tokenize, &bpe_module)?)?;
    bpe_module.add_function(wrap_pyfunction!(bpe::merge::merge, &bpe_module)?)?;
    bpe_module.add_function(wrap_pyfunction!(bpe::train, &bpe_module)?)?;

    // 3. 将子模块添加到父模块 (rust_lib)
    m.add_submodule(&bpe_module)?;

    Ok(())
}
