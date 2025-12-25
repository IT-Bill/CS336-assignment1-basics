use pyo3::prelude::*;

mod pre_tokenize;
mod merge;

/// A Python module implemented in Rust.
#[pymodule]
fn rust_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pre_tokenize::pre_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(merge::merge, m)?)?;
    Ok(())
}
