#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

mod prelude;
pub(crate) use prelude::*;

mod common;
pub(crate) use common::*;
mod checkpoint;
pub(crate) use checkpoint::*;
mod stats;
pub(crate) use stats::*;
mod autotune;
pub(crate) use autotune::*;
mod data_loader;
pub(crate) use data_loader::*;
mod distributed;
pub(crate) use distributed::*;
mod image_loader;
pub(crate) use image_loader::*;
mod video_loader;
pub(crate) use video_loader::*;
mod text_loader;
pub(crate) use text_loader::*;
mod audio_loader;
pub(crate) use audio_loader::*;
mod mix_loader;
pub(crate) use mix_loader::*;
mod api;
pub(crate) use api::*;

#[pymodule]
fn mx8(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let internal = PyModule::new_bound(py, "_internal")?;
    internal.add_function(wrap_pyfunction!(internal_video_index_build, &internal)?)?;
    internal.add_function(wrap_pyfunction!(
        internal_video_index_replay_check,
        &internal
    )?)?;
    m.add_submodule(&internal)?;
    m.setattr("_internal", &internal)?;

    m.add_class::<DataLoader>()?;
    m.add_class::<TextLoader>()?;
    m.add_class::<AudioLoader>()?;
    m.add_class::<ImageLoader>()?;
    m.add_class::<MixedDataLoader>()?;
    m.add_class::<VideoDataLoader>()?;
    m.add_class::<VideoBatch>()?;
    m.add_class::<DistributedDataLoader>()?;
    m.add_class::<Constraints>()?;
    m.add_class::<RuntimeConfig>()?;
    m.add_class::<PyBatch>()?;
    m.add_class::<TextBatch>()?;
    m.add_class::<AudioBatch>()?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add_function(wrap_pyfunction!(pack_dir, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(py_text, m)?)?;
    m.add_function(wrap_pyfunction!(py_audio, m)?)?;
    m.add_function(wrap_pyfunction!(py_image, m)?)?;
    m.add_function(wrap_pyfunction!(video, m)?)?;
    m.add_function(wrap_pyfunction!(mix, m)?)?;
    m.add_function(wrap_pyfunction!(api::stats, m)?)?;
    m.add_function(wrap_pyfunction!(resolve, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_manifest_hash, m)?)?;
    m.add_class::<CoordinatorHandle>()?;
    m.add_function(wrap_pyfunction!(coordinator, m)?)?;
    Ok(())
}
