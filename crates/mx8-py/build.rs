fn main() {
    // On macOS, Python extension modules are loaded into the Python process and
    // resolve CPython symbols at runtime from the host interpreter.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
