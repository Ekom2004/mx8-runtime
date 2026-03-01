use super::*;

pub(crate) struct DecodeBatchResult {
    nchw_u8: Vec<u8>,
    h: usize,
    w: usize,
    decode_ms: u64,
    resize_ms: u64,
    pack_ms: u64,
}

pub(crate) enum ImageLoaderInner {
    Local(DataLoader),
    Distributed(DistributedDataLoader),
}

#[pyclass]
pub(crate) struct ImageLoader {
    pub(crate) loader: ImageLoaderInner,
    pub(crate) resize_hw: Option<(u32, u32)>,
    pub(crate) crop_hw: Option<(u32, u32)>,
    pub(crate) horizontal_flip_p: f32,
    pub(crate) color_jitter_brightness: f32,
    pub(crate) color_jitter_contrast: f32,
    pub(crate) color_jitter_saturation: f32,
    pub(crate) color_jitter_hue: f32,
    pub(crate) normalize_mean: Option<[f32; 3]>,
    pub(crate) normalize_std: Option<[f32; 3]>,
    pub(crate) seed: u64,
    pub(crate) epoch: u64,
    pub(crate) manifest_hash_seed: u64,
    pub(crate) to_float: bool,
    pub(crate) decode_backend: DecodeBackend,
    pub(crate) rust_jpeg_codec: RustJpegCodec,
    pub(crate) rust_resize_backend: RustResizeBackend,
    pub(crate) decode_threads: usize,
    pub(crate) decode_pool: Option<Arc<rayon::ThreadPool>>,
    pub(crate) classes: Option<Vec<String>>,
}

pub(crate) fn stable_hash64(bytes: &[u8]) -> u64 {
    // FNV-1a 64-bit for deterministic per-run hashing.
    let mut hash = 0xcbf29ce484222325u64;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub(crate) fn unit_f32(
    manifest_hash_seed: u64,
    seed: u64,
    sample_id: u64,
    stream_id: u64,
    epoch: u64,
) -> f32 {
    let x = manifest_hash_seed
        .wrapping_add(seed.rotate_left(7))
        .wrapping_add(sample_id.rotate_left(17))
        .wrapping_add(stream_id.rotate_left(29))
        .wrapping_add(epoch.rotate_left(41));
    let bits = splitmix64(x);
    // Map to [0, 1).
    (bits as f64 / (u64::MAX as f64 + 1.0)) as f32
}

pub(crate) fn decode_images_nchw_u8(
    lease: &BatchLease,
    resize_hw: Option<(u32, u32)>,
    rust_jpeg_codec: RustJpegCodec,
    rust_resize_backend: RustResizeBackend,
    decode_threads: usize,
    decode_pool: Option<&rayon::ThreadPool>,
) -> PyResult<DecodeBatchResult> {
    struct DecodedRgb {
        sample_id: u64,
        h: usize,
        w: usize,
        rgb_u8: Vec<u8>,
    }

    struct DecodeOneResult {
        image: DecodedRgb,
        decode_us: u64,
        resize_us: u64,
    }

    fn elapsed_us_u64(started: Instant) -> u64 {
        let micros = started.elapsed().as_micros();
        if micros > u128::from(u64::MAX) {
            u64::MAX
        } else {
            micros as u64
        }
    }

    fn decode_one_image(
        sample_id: u64,
        bytes: &[u8],
        resize_hw: Option<(u32, u32)>,
        rust_jpeg_codec: RustJpegCodec,
        rust_resize_backend: RustResizeBackend,
    ) -> Result<DecodeOneResult, String> {
        fn looks_like_jpeg(bytes: &[u8]) -> bool {
            bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8
        }

        fn decode_rgb_with_image(
            bytes: &[u8],
            sample_id: u64,
        ) -> Result<(Vec<u8>, u32, u32), String> {
            let decoded = ::image::load_from_memory(bytes)
                .map_err(|e| format!("decode failed for sample_id {sample_id}: {e}"))?;
            let rgb = decoded.to_rgb8();
            let width = rgb.width();
            let height = rgb.height();
            Ok((rgb.into_raw(), width, height))
        }

        fn decode_jpeg_rgb_with_zune(
            bytes: &[u8],
            sample_id: u64,
        ) -> Result<(Vec<u8>, u32, u32), String> {
            let cursor = ZCursor::new(bytes);
            let mut decoder = JpegDecoder::new(cursor);
            let pixels = decoder
                .decode()
                .map_err(|e| format!("zune decode failed for sample_id {sample_id}: {e}"))?;
            let info = decoder.info().ok_or_else(|| {
                format!("zune decode missing image info for sample_id {sample_id}")
            })?;
            let width = u32::from(info.width);
            let height = u32::from(info.height);
            let expected_len = usize::try_from(width)
                .ok()
                .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
                .and_then(|px| px.checked_mul(3))
                .ok_or_else(|| format!("zune decoded shape overflow for sample_id {sample_id}"))?;
            if pixels.len() != expected_len {
                return Err(format!(
                    "zune decoded unexpected channel count for sample_id {sample_id} (len={}, expected_rgb_len={expected_len})",
                    pixels.len()
                ));
            }
            Ok((pixels, width, height))
        }

        fn decode_jpeg_rgb_with_turbo(
            bytes: &[u8],
            sample_id: u64,
        ) -> Result<(Vec<u8>, u32, u32), String> {
            let decoded = turbojpeg::decompress(bytes, TurboPixelFormat::RGB)
                .map_err(|e| format!("turbojpeg decode failed for sample_id {sample_id}: {e}"))?;
            let width_u32 = u32::try_from(decoded.width)
                .map_err(|_| format!("turbojpeg width overflow for sample_id {sample_id}"))?;
            let height_u32 = u32::try_from(decoded.height)
                .map_err(|_| format!("turbojpeg height overflow for sample_id {sample_id}"))?;
            let row_bytes = decoded
                .width
                .checked_mul(3)
                .ok_or_else(|| format!("turbojpeg row size overflow for sample_id {sample_id}"))?;
            let expected_len = decoded
                .width
                .checked_mul(decoded.height)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| {
                    format!("turbojpeg decoded shape overflow for sample_id {sample_id}")
                })?;

            if decoded.pitch == row_bytes && decoded.pixels.len() == expected_len {
                return Ok((decoded.pixels, width_u32, height_u32));
            }

            if decoded.pitch < row_bytes {
                return Err(format!(
                    "turbojpeg invalid pitch for sample_id {sample_id} (pitch={} row_bytes={row_bytes})",
                    decoded.pitch
                ));
            }
            let required_len = decoded.height.checked_mul(decoded.pitch).ok_or_else(|| {
                format!("turbojpeg pitch buffer overflow for sample_id {sample_id}")
            })?;
            if decoded.pixels.len() < required_len {
                return Err(format!(
                    "turbojpeg decoded buffer too small for sample_id {sample_id} (len={}, required={required_len})",
                    decoded.pixels.len()
                ));
            }

            let mut compact = vec![0u8; expected_len];
            for y in 0..decoded.height {
                let src_start = y.checked_mul(decoded.pitch).ok_or_else(|| {
                    format!("turbojpeg src offset overflow for sample_id {sample_id}")
                })?;
                let src_end = src_start.checked_add(row_bytes).ok_or_else(|| {
                    format!("turbojpeg src end overflow for sample_id {sample_id}")
                })?;
                let dst_start = y.checked_mul(row_bytes).ok_or_else(|| {
                    format!("turbojpeg dst offset overflow for sample_id {sample_id}")
                })?;
                let dst_end = dst_start.checked_add(row_bytes).ok_or_else(|| {
                    format!("turbojpeg dst end overflow for sample_id {sample_id}")
                })?;
                compact[dst_start..dst_end].copy_from_slice(&decoded.pixels[src_start..src_end]);
            }
            Ok((compact, width_u32, height_u32))
        }

        let decode_started = Instant::now();
        let (raw_rgb, width, height) = match rust_jpeg_codec {
            RustJpegCodec::Zune if looks_like_jpeg(bytes) => {
                decode_jpeg_rgb_with_zune(bytes, sample_id)?
            }
            RustJpegCodec::Turbo if looks_like_jpeg(bytes) => {
                decode_jpeg_rgb_with_turbo(bytes, sample_id)?
            }
            _ => decode_rgb_with_image(bytes, sample_id)?,
        };
        let decode_us = elapsed_us_u64(decode_started);

        let (raw_rgb, width, height, resize_us) = if let Some((h, w)) = resize_hw {
            let resize_started = Instant::now();
            let resized = match rust_resize_backend {
                RustResizeBackend::FastImageResize => {
                    let src_image =
                        FirImage::from_vec_u8(width, height, raw_rgb, FirPixelType::U8x3).map_err(
                            |e| {
                                format!(
                                    "fast resize source init failed for sample_id {sample_id}: {e}"
                                )
                            },
                        )?;
                    let mut dst_image = FirImage::new(w, h, FirPixelType::U8x3);
                    let mut resizer = FirResizer::new();
                    let resize_options = FirResizeOptions::new()
                        .resize_alg(FirResizeAlg::Convolution(FirFilterType::Bilinear));
                    resizer
                        .resize(&src_image, &mut dst_image, &resize_options)
                        .map_err(|e| {
                            format!("fast resize failed for sample_id {sample_id}: {e}")
                        })?;
                    dst_image.into_vec()
                }
                RustResizeBackend::Image => {
                    let rgb =
                        ::image::RgbImage::from_raw(width, height, raw_rgb).ok_or_else(|| {
                            format!("decoded rgb buffer shape mismatch for sample_id {sample_id}")
                        })?;
                    let resized = ::image::imageops::resize(&rgb, w, h, ImageFilterType::Triangle);
                    resized.into_raw()
                }
            };
            (resized, w, h, elapsed_us_u64(resize_started))
        } else {
            (raw_rgb, width, height, 0)
        };

        let h =
            usize::try_from(height).map_err(|_| "image height does not fit usize".to_string())?;
        let w = usize::try_from(width).map_err(|_| "image width does not fit usize".to_string())?;
        let rgb_len = h
            .checked_mul(w)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| "decoded rgb size overflow".to_string())?;
        if raw_rgb.len() != rgb_len {
            return Err(format!(
                "decoded rgb length mismatch for sample_id {sample_id} (len={}, expected={rgb_len})",
                raw_rgb.len()
            ));
        }
        Ok(DecodeOneResult {
            image: DecodedRgb {
                sample_id,
                h,
                w,
                rgb_u8: raw_rgb,
            },
            decode_us,
            resize_us,
        })
    }

    let sample_ids = lease.batch.sample_ids.clone();
    let offsets = lease.batch.offsets.clone();
    let payload = lease.batch.payload.clone();
    let sample_count = sample_ids.len();
    if sample_count == 0 {
        return Err(PyRuntimeError::new_err("empty image batch"));
    }

    let decode_us_total = Arc::new(AtomicU64::new(0));
    let resize_us_total = Arc::new(AtomicU64::new(0));

    let decode_at = |i: usize| -> Result<DecodedRgb, String> {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        if end < start || end > payload.len() {
            return Err(format!(
                "bad offsets for sample_id {} (start={} end={} payload_len={})",
                sample_ids[i],
                start,
                end,
                payload.len()
            ));
        }
        let one = decode_one_image(
            sample_ids[i],
            &payload[start..end],
            resize_hw,
            rust_jpeg_codec,
            rust_resize_backend,
        )?;
        decode_us_total.fetch_add(one.decode_us, Ordering::Relaxed);
        resize_us_total.fetch_add(one.resize_us, Ordering::Relaxed);
        Ok(one.image)
    };

    let decoded: Vec<Result<DecodedRgb, String>> =
        if decode_threads > 1 && sample_count > 1 && decode_pool.is_some() {
            decode_pool
                .ok_or_else(|| PyRuntimeError::new_err("decode thread pool unavailable"))?
                .install(|| (0..sample_count).into_par_iter().map(decode_at).collect())
        } else {
            (0..sample_count).map(decode_at).collect()
        };

    let mut images = Vec::with_capacity(sample_count);
    let mut first_h: Option<usize> = None;
    let mut first_w: Option<usize> = None;
    for maybe_image in decoded {
        let image = maybe_image.map_err(PyRuntimeError::new_err)?;

        match (first_h, first_w) {
            (None, None) => {
                first_h = Some(image.h);
                first_w = Some(image.w);
            }
            (Some(h0), Some(w0)) if h0 == image.h && w0 == image.w => {}
            (Some(h0), Some(w0)) => {
                return Err(PyValueError::new_err(format!(
                    "decoded image shape mismatch in batch (sample_id={} got={}x{}, expected={}x{}); set resize_hw for variable-size inputs",
                    image.sample_id, image.h, image.w, h0, w0
                )));
            }
            _ => {}
        }
        images.push(image);
    }

    let h = first_h.ok_or_else(|| PyRuntimeError::new_err("empty image batch"))?;
    let w = first_w.ok_or_else(|| PyRuntimeError::new_err("empty image batch"))?;
    let pixels_per_image = h
        .checked_mul(w)
        .ok_or_else(|| PyRuntimeError::new_err("pixels_per_image overflow"))?;
    let bytes_per_image = pixels_per_image
        .checked_mul(3)
        .ok_or_else(|| PyRuntimeError::new_err("bytes_per_image overflow"))?;
    let total_bytes = sample_count
        .checked_mul(bytes_per_image)
        .ok_or_else(|| PyRuntimeError::new_err("decoded batch byte size overflow"))?;
    let mut out = vec![0u8; total_bytes];

    let pack_started = Instant::now();
    for (image_index, image) in images.iter().enumerate() {
        let image_base = image_index
            .checked_mul(bytes_per_image)
            .ok_or_else(|| PyRuntimeError::new_err("decoded batch index overflow"))?;
        let (c0, tail) =
            out[image_base..image_base + bytes_per_image].split_at_mut(pixels_per_image);
        let (c1, c2) = tail.split_at_mut(pixels_per_image);
        for pixel_idx in 0..pixels_per_image {
            let src = pixel_idx
                .checked_mul(3)
                .ok_or_else(|| PyRuntimeError::new_err("decoded pixel index overflow"))?;
            c0[pixel_idx] = image.rgb_u8[src];
            c1[pixel_idx] = image.rgb_u8[src + 1];
            c2[pixel_idx] = image.rgb_u8[src + 2];
        }
    }

    Ok(DecodeBatchResult {
        nchw_u8: out,
        h,
        w,
        decode_ms: decode_us_total.load(Ordering::Relaxed) / 1000,
        resize_ms: resize_us_total.load(Ordering::Relaxed) / 1000,
        pack_ms: elapsed_us_u64(pack_started) / 1000,
    })
}

impl ImageLoader {
    fn apply_image_augmentations<'py>(
        &self,
        py: Python<'py>,
        torch: &Bound<'py, PyModule>,
        images: Bound<'py, PyAny>,
        sample_ids: &[u64],
    ) -> PyResult<Bound<'py, PyAny>> {
        let augment_enabled = self.crop_hw.is_some()
            || self.horizontal_flip_p > 0.0
            || self.color_jitter_brightness > 0.0
            || self.color_jitter_contrast > 0.0
            || self.color_jitter_saturation > 0.0
            || self.color_jitter_hue > 0.0
            || (self.normalize_mean.is_some() && self.normalize_std.is_some());
        if !augment_enabled || sample_ids.is_empty() {
            return Ok(images);
        }
        if self.color_jitter_hue > 0.0 {
            return Err(PyValueError::new_err(
                "color_jitter_hue > 0 is not supported yet (set hue to 0.0)",
            ));
        }

        let shape_obj = images.getattr("shape")?;
        let shape = shape_obj.downcast::<PyTuple>()?;
        if shape.len() != 4 {
            return Err(PyRuntimeError::new_err(
                "expected images tensor shape [B,C,H,W] for augmentations",
            ));
        }
        let src_h = shape.get_item(2)?.extract::<i64>()?;
        let src_w = shape.get_item(3)?.extract::<i64>()?;
        let mut crop_h_i64 = src_h;
        let mut crop_w_i64 = src_w;
        if let Some((crop_h, crop_w)) = self.crop_hw {
            crop_h_i64 = i64::from(crop_h);
            crop_w_i64 = i64::from(crop_w);
            if crop_h_i64 <= 0 || crop_w_i64 <= 0 {
                return Err(PyValueError::new_err("crop_hw must be positive"));
            }
            if crop_h_i64 > src_h || crop_w_i64 > src_w {
                return Err(PyValueError::new_err(format!(
                    "crop_hw {:?} exceeds image shape {}x{}",
                    self.crop_hw, src_h, src_w
                )));
            }
        }

        let mut out_samples: Vec<PyObject> = Vec::with_capacity(sample_ids.len());
        for (i, sample_id) in sample_ids.iter().enumerate() {
            let mut sample = images.call_method1("__getitem__", (i as i64,))?;

            if self.crop_hw.is_some() {
                let top_max = src_h - crop_h_i64;
                let left_max = src_w - crop_w_i64;
                let top = if top_max > 0 {
                    let u = unit_f32(
                        self.manifest_hash_seed,
                        self.seed,
                        *sample_id,
                        1,
                        self.epoch,
                    );
                    (((top_max + 1) as f32) * u)
                        .floor()
                        .clamp(0.0, top_max as f32) as i64
                } else {
                    0
                };
                let left = if left_max > 0 {
                    let u = unit_f32(
                        self.manifest_hash_seed,
                        self.seed,
                        *sample_id,
                        2,
                        self.epoch,
                    );
                    (((left_max + 1) as f32) * u)
                        .floor()
                        .clamp(0.0, left_max as f32) as i64
                } else {
                    0
                };
                sample = sample
                    .call_method1("narrow", (1i64, top, crop_h_i64))?
                    .call_method1("narrow", (2i64, left, crop_w_i64))?;
            }

            if self.horizontal_flip_p > 0.0 {
                let u = unit_f32(
                    self.manifest_hash_seed,
                    self.seed,
                    *sample_id,
                    3,
                    self.epoch,
                );
                if u < self.horizontal_flip_p {
                    let dims = PyTuple::new_bound(py, [2i64]);
                    sample = sample.call_method1("flip", (dims,))?;
                }
            }

            if self.color_jitter_brightness > 0.0
                || self.color_jitter_contrast > 0.0
                || self.color_jitter_saturation > 0.0
            {
                sample = sample.call_method0("float")?;
                sample = sample.call_method1("clamp", (0.0f32, 1.0f32))?;
            }

            if self.color_jitter_brightness > 0.0 {
                let u = unit_f32(
                    self.manifest_hash_seed,
                    self.seed,
                    *sample_id,
                    11,
                    self.epoch,
                );
                let factor = 1.0 + (u * 2.0 - 1.0) * self.color_jitter_brightness;
                sample = sample
                    .call_method1("mul", (factor,))?
                    .call_method1("clamp", (0.0f32, 1.0f32))?;
            }

            if self.color_jitter_contrast > 0.0 {
                let u = unit_f32(
                    self.manifest_hash_seed,
                    self.seed,
                    *sample_id,
                    12,
                    self.epoch,
                );
                let factor = 1.0 + (u * 2.0 - 1.0) * self.color_jitter_contrast;
                let mean = sample.call_method0("mean")?;
                sample = sample
                    .call_method1("sub", (mean.clone(),))?
                    .call_method1("mul", (factor,))?
                    .call_method1("add", (mean,))?
                    .call_method1("clamp", (0.0f32, 1.0f32))?;
            }

            if self.color_jitter_saturation > 0.0 {
                let u = unit_f32(
                    self.manifest_hash_seed,
                    self.seed,
                    *sample_id,
                    13,
                    self.epoch,
                );
                let factor = 1.0 + (u * 2.0 - 1.0) * self.color_jitter_saturation;
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("dim", PyTuple::new_bound(py, [0i64]))?;
                kwargs.set_item("keepdim", true)?;
                let gray = sample.call_method("mean", (), Some(&kwargs))?;
                sample = sample
                    .call_method1("sub", (gray.clone(),))?
                    .call_method1("mul", (factor,))?
                    .call_method1("add", (gray,))?
                    .call_method1("clamp", (0.0f32, 1.0f32))?;
            }

            out_samples.push(sample.to_object(py));
        }

        let xs_list = PyList::new_bound(py, out_samples);
        let mut out = torch.getattr("stack")?.call1((xs_list, 0i64))?;
        if let (Some(mean), Some(std)) = (self.normalize_mean, self.normalize_std) {
            out = out.call_method0("float")?;
            let dtype = out.getattr("dtype")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", dtype)?;

            let mean_values = PyList::new_bound(py, mean);
            let std_values = PyList::new_bound(py, std);
            let mean_t = torch
                .call_method("tensor", (mean_values,), Some(&kwargs))?
                .call_method1("view", (1i64, 3i64, 1i64, 1i64))?;
            let std_t = torch
                .call_method("tensor", (std_values,), Some(&kwargs))?
                .call_method1("view", (1i64, 3i64, 1i64, 1i64))?;
            out = out
                .call_method1("sub", (mean_t,))?
                .call_method1("div", (std_t,))?;
        }
        Ok(out)
    }

    fn next_python_decode<'py>(
        &self,
        py: Python<'py>,
        lease: BatchLease,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sample_count = lease.batch.sample_count();
        let started = std::time::Instant::now();
        let labels = lease.batch.label_ids.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "batch has no label_ids (expected mx8:vision:imagefolder label hints in manifest)",
            )
        })?;

        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use ImageLoader): {e}"
            ))
        })?;
        let np = py.import_bound("numpy").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import numpy (install numpy to use ImageLoader): {e}"
            ))
        })?;
        let pil = py.import_bound("PIL.Image").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import PIL.Image (install Pillow to use ImageLoader): {e}"
            ))
        })?;
        let io = py.import_bound("io")?;

        let bytes_io = io.getattr("BytesIO")?;
        let image_open = pil.getattr("open")?;

        let mut xs: Vec<Bound<'py, PyAny>> = Vec::with_capacity(lease.batch.sample_count());

        for i in 0..lease.batch.sample_count() {
            let start = lease.batch.offsets[i] as usize;
            let end = lease.batch.offsets[i + 1] as usize;
            if end < start || end > lease.batch.payload.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "bad offsets for sample_id {} (start={} end={} payload_len={})",
                    lease.batch.sample_ids[i],
                    start,
                    end,
                    lease.batch.payload.len()
                )));
            }

            let b = PyBytes::new_bound(py, &lease.batch.payload[start..end]);
            let bio = bytes_io.call1((b,))?;

            let mut img = image_open
                .call1((bio,))?
                .call_method1("convert", ("RGB",))?;
            if let Some((h, w)) = self.resize_hw {
                img = img.call_method1("resize", ((w, h),))?;
            }

            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", np.getattr("uint8")?)?;
            kwargs.set_item("copy", true)?;
            let arr = np.call_method("array", (img,), Some(&kwargs))?;

            let x = torch.getattr("from_numpy")?.call1((arr,))?;
            let x = x
                .call_method1("permute", (2i64, 0i64, 1i64))?
                .call_method0("contiguous")?;
            let x = if self.to_float {
                x.call_method0("float")?.call_method1("div", (255.0f32,))?
            } else {
                x
            };
            xs.push(x);
        }

        let xs_list = PyList::new_bound(py, xs);
        let images = torch.getattr("stack")?.call1((xs_list, 0i64))?;
        let images = self.apply_image_augmentations(py, &torch, images, &lease.batch.sample_ids)?;
        let labels = labels_to_torch_i64(py, labels)?;
        let out = PyTuple::new_bound(py, [images.to_object(py), labels.to_object(py)]);
        let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        tracing::debug!(
            target: "mx8_proof",
            event = "vision_decode_batch",
            backend = "python",
            samples = sample_count as u64,
            elapsed_ms = elapsed_ms,
            "vision decode batch"
        );
        Ok(out.into_any())
    }

    fn next_rust_decode<'py>(
        &self,
        py: Python<'py>,
        lease: BatchLease,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sample_count = lease.batch.sample_count();
        let started = std::time::Instant::now();
        let labels = lease.batch.label_ids.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "batch has no label_ids (expected mx8:vision:imagefolder label hints in manifest)",
            )
        })?;

        let decode_result = py.allow_threads(|| {
            decode_images_nchw_u8(
                &lease,
                self.resize_hw,
                self.rust_jpeg_codec,
                self.rust_resize_backend,
                self.decode_threads,
                self.decode_pool.as_deref(),
            )
        })?;

        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use ImageLoader): {e}"
            ))
        })?;
        let torch_uint8 = torch.getattr("uint8")?;

        let payload = PyByteArray::new_bound(py, &decode_result.nchw_u8);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_uint8)?;
        let images = torch.call_method("frombuffer", (payload,), Some(&kwargs))?;

        let b_i64 = i64::try_from(lease.batch.sample_count())
            .map_err(|_| PyValueError::new_err("batch size does not fit i64"))?;
        let h_i64 = i64::try_from(decode_result.h)
            .map_err(|_| PyValueError::new_err("height does not fit i64"))?;
        let w_i64 = i64::try_from(decode_result.w)
            .map_err(|_| PyValueError::new_err("width does not fit i64"))?;

        let images = images
            .call_method1("view", (b_i64, 3i64, h_i64, w_i64))?
            .call_method0("contiguous")?;
        let images = if self.to_float {
            images
                .call_method0("float")?
                .call_method1("div", (255.0f32,))?
        } else {
            images
        };
        let images = self.apply_image_augmentations(py, &torch, images, &lease.batch.sample_ids)?;

        let labels = labels_to_torch_i64(py, labels)?;
        let out = PyTuple::new_bound(py, [images.to_object(py), labels.to_object(py)]);
        let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        tracing::debug!(
            target: "mx8_proof",
            event = "vision_decode_batch",
            backend = "rust",
            samples = sample_count as u64,
            elapsed_ms = elapsed_ms,
            decode_ms = decode_result.decode_ms,
            resize_ms = decode_result.resize_ms,
            pack_ms = decode_result.pack_ms,
            decode_threads = self.decode_threads as u64,
            rust_jpeg_codec = rust_jpeg_codec_name(self.rust_jpeg_codec),
            rust_resize_backend = rust_resize_backend_name(self.rust_resize_backend),
            "vision decode batch"
        );
        Ok(out.into_any())
    }
}

#[pymethods]
impl ImageLoader {
    #[new]
    #[pyo3(signature = (
        dataset_link,
        *,
        manifest_store=None,
        manifest_path=None,
        recursive=true,
        batch_size_samples=32,
        max_inflight_bytes=128*1024*1024,
        max_queue_batches=64,
        prefetch_batches=1,
        target_batch_bytes=None,
        max_batch_bytes=None,
        max_ram_bytes=None,
        start_id=None,
        end_id=None,
        resume_from=None,
        node_id=None,
        profile=None,
        autotune=None,
        resize_hw=None,
        crop_hw=None,
        horizontal_flip_p=0.0,
        color_jitter_brightness=0.0,
        color_jitter_contrast=0.0,
        color_jitter_saturation=0.0,
        color_jitter_hue=0.0,
        normalize_mean=None,
        normalize_std=None,
        seed=0,
        epoch=0,
        to_float=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dataset_link: String,
        manifest_store: Option<PathBuf>,
        manifest_path: Option<PathBuf>,
        recursive: bool,
        batch_size_samples: usize,
        max_inflight_bytes: u64,
        max_queue_batches: usize,
        prefetch_batches: usize,
        target_batch_bytes: Option<u64>,
        max_batch_bytes: Option<u64>,
        max_ram_bytes: Option<u64>,
        start_id: Option<u64>,
        end_id: Option<u64>,
        resume_from: Option<Vec<u8>>,
        node_id: Option<String>,
        profile: Option<String>,
        autotune: Option<bool>,
        resize_hw: Option<(u32, u32)>,
        crop_hw: Option<(u32, u32)>,
        horizontal_flip_p: f32,
        color_jitter_brightness: f32,
        color_jitter_contrast: f32,
        color_jitter_saturation: f32,
        color_jitter_hue: f32,
        normalize_mean: Option<(f32, f32, f32)>,
        normalize_std: Option<(f32, f32, f32)>,
        seed: u64,
        epoch: u64,
        to_float: bool,
    ) -> PyResult<Self> {
        let loader = DataLoader::new(
            dataset_link,
            manifest_store,
            manifest_path,
            recursive,
            batch_size_samples,
            max_inflight_bytes,
            max_queue_batches,
            prefetch_batches,
            target_batch_bytes,
            max_batch_bytes,
            max_ram_bytes,
            start_id,
            end_id,
            resume_from,
            node_id,
            profile,
            autotune,
        )?;
        let base = loader.dataset_base.clone();
        let classes = loader
            .rt
            .block_on(async { load_labels_for_base(&base).await })
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        let decode_backend = decode_backend_from_env()?;
        let rust_jpeg_codec = rust_jpeg_codec_from_env()?;
        let rust_resize_backend = rust_resize_backend_from_env()?;
        let decode_threads = decode_threads_from_env()?;
        let decode_pool = if matches!(decode_backend, DecodeBackend::Rust) && decode_threads > 1 {
            Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(decode_threads)
                    .build()
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "failed to build rust decode thread pool: {e}"
                        ))
                    })?,
            ))
        } else {
            None
        };
        tracing::info!(
            target: "mx8_proof",
            event = "vision_decode_backend_selected",
            backend = decode_backend_name(decode_backend),
            decode_threads = decode_threads,
            rust_jpeg_codec = rust_jpeg_codec_name(rust_jpeg_codec),
            rust_resize_backend = rust_resize_backend_name(rust_resize_backend),
            "vision decode backend selected"
        );
        let normalize_mean = normalize_mean.map(|(r, g, b)| [r, g, b]);
        let normalize_std = normalize_std.map(|(r, g, b)| [r, g, b]);
        let manifest_hash_seed = stable_hash64(loader.manifest_hash.as_bytes());
        Ok(Self {
            loader: ImageLoaderInner::Local(loader),
            resize_hw,
            crop_hw,
            horizontal_flip_p,
            color_jitter_brightness,
            color_jitter_contrast,
            color_jitter_saturation,
            color_jitter_hue,
            normalize_mean,
            normalize_std,
            seed,
            epoch,
            manifest_hash_seed,
            to_float,
            decode_backend,
            rust_jpeg_codec,
            rust_resize_backend,
            decode_threads,
            decode_pool,
            classes,
        })
    }

    #[getter]
    fn classes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.classes {
            Some(v) => Ok(PyList::new_bound(py, v).into_any()),
            None => Ok(py.None().into_bound(py).into_any()),
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.loader {
            ImageLoaderInner::Local(loader) => loader.stats(py),
            ImageLoaderInner::Distributed(loader) => loader.stats(py),
        }
    }

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match &self.loader {
            ImageLoaderInner::Local(loader) => loader.checkpoint(py),
            ImageLoaderInner::Distributed(loader) => loader.checkpoint(py),
        }
    }

    fn print_stats(&self, py: Python<'_>) -> PyResult<()> {
        match &self.loader {
            ImageLoaderInner::Local(loader) => loader.print_stats(py),
            ImageLoaderInner::Distributed(loader) => {
                let stats = loader.stats(py)?;
                let stats = stats.downcast::<PyDict>()?;
                let text = render_human_stats(stats).replace('\n', " | ");
                eprintln!("[mx8] {text}");
                Ok(())
            }
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let lease = match &mut self.loader {
            ImageLoaderInner::Local(loader) => {
                let lease =
                    py.allow_threads(|| loader.rt.block_on(async { loader.rx.recv().await }));
                let Some(lease) = lease else {
                    let Some(task) = loader.task.take() else {
                        return Err(PyStopIteration::new_err(()));
                    };
                    let out = py.allow_threads(|| loader.rt.block_on(task));
                    return match out {
                        Ok(Ok(())) => Err(PyStopIteration::new_err(())),
                        Ok(Err(err)) => Err(PyRuntimeError::new_err(format!("{err}"))),
                        Err(err) => Err(PyRuntimeError::new_err(format!(
                            "producer task failed: {err}"
                        ))),
                    };
                };
                loader.on_batch_delivered(&lease);
                lease
            }
            ImageLoaderInner::Distributed(loader) => {
                let wait_started = Instant::now();
                let lease =
                    py.allow_threads(|| loader.rt.block_on(async { loader.rx.recv().await }));
                let Some(lease) = lease else {
                    let Some(task) = loader.task.take() else {
                        return Err(PyStopIteration::new_err(()));
                    };
                    let out = py.allow_threads(|| loader.rt.block_on(task));
                    return match out {
                        Ok(Ok(())) => Err(PyStopIteration::new_err(())),
                        Ok(Err(err)) => Err(PyRuntimeError::new_err(format!("{err}"))),
                        Err(err) => Err(PyRuntimeError::new_err(format!(
                            "producer task failed: {err}"
                        ))),
                    };
                };
                loader.autotune.on_wait(wait_started.elapsed());
                lease
            }
        };

        match self.decode_backend {
            DecodeBackend::Rust => self.next_rust_decode(py, lease),
            DecodeBackend::Python => self.next_python_decode(py, lease),
        }
    }

    fn close(&mut self) {
        match &mut self.loader {
            ImageLoaderInner::Local(loader) => loader.close(),
            ImageLoaderInner::Distributed(loader) => loader.close(),
        }
    }

    fn __del__(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod image_aug_tests {
    use super::{stable_hash64, unit_f32};

    #[test]
    fn image_aug_rng_is_deterministic_for_fixed_inputs() {
        let manifest = stable_hash64(b"manifest-abc");
        let a = unit_f32(manifest, 7, 42, 3, 0);
        let b = unit_f32(manifest, 7, 42, 3, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn image_aug_rng_changes_with_epoch() {
        let manifest = stable_hash64(b"manifest-abc");
        let epoch0 = unit_f32(manifest, 7, 42, 3, 0);
        let epoch1 = unit_f32(manifest, 7, 42, 3, 1);
        assert_ne!(epoch0, epoch1);
    }

    #[test]
    fn stable_hash64_is_stable() {
        assert_eq!(stable_hash64(b"mx8"), 0x07ba8b1917254288);
    }
}
