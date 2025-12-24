use pyo3::prelude::*;

mod pre_tokenize;

/// A Python module implemented in Rust.
#[pymodule]
fn rust_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pre_tokenize::pre_tokenize, m)?)?;
    Ok(())
}
