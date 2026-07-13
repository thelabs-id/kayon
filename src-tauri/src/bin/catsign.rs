//! catsign — sign the bundled catalog with the project's ed25519 signing key (CAT-5/CAT-6).
//!
//! The private signing key is NOT stored in source. It is read from (in order):
//!   1. `KAYON_CATALOG_SEED` — 64-hex-char (32-byte) seed in the environment (CI secret / HSM export)
//!   2. a key file at `KAYON_CATALOG_SEED_FILE` (default: `catalog/signing.key`, gitignored)
//! If neither exists, `sign`/`pubkey` generate a fresh random key, persist it to the key file, and
//! print the new verifying key to bake into `catalog::BAKED_PUBKEY`. Rotating the key therefore
//! means: run `catsign pubkey`, paste the new key, run `catsign sign`.
//!
//! Usage:
//!   cargo run --bin catsign -- pubkey    # print the [u8;32] verifying-key literal
//!   cargo run --bin catsign -- sign      # write catalog/catalog.json.sig

use ed25519_dalek::{Signer, SigningKey};
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn key_file() -> PathBuf {
    std::env::var("KAYON_CATALOG_SEED_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| crate_root().join("catalog").join("signing.key"))
}

/// Load the signing key from env or the key file; generate + persist one if absent.
fn signing_key() -> SigningKey {
    if let Ok(hex_seed) = std::env::var("KAYON_CATALOG_SEED") {
        let bytes = hex::decode(hex_seed.trim()).expect("KAYON_CATALOG_SEED must be hex");
        let seed: [u8; 32] = bytes.as_slice().try_into().expect("seed must be 32 bytes");
        return SigningKey::from_bytes(&seed);
    }
    let path = key_file();
    if let Ok(bytes) = std::fs::read(&path) {
        let seed: [u8; 32] = bytes.as_slice().try_into().expect("key file must be 32 bytes");
        return SigningKey::from_bytes(&seed);
    }
    // No key anywhere yet: generate one and persist it to the (gitignored) key file.
    use rand::RngCore;
    let mut seed = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut seed);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&path, seed).expect("persist signing key");
    eprintln!("generated a new signing key at {}", path.display());
    SigningKey::from_bytes(&seed)
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "sign".to_string());
    let sk = signing_key();
    let vk = sk.verifying_key();

    match mode.as_str() {
        "pubkey" => {
            print!("pub const BAKED_PUBKEY: [u8; 32] = [");
            for (i, b) in vk.to_bytes().iter().enumerate() {
                if i % 8 == 0 {
                    print!("\n    ");
                }
                print!("0x{:02x}, ", b);
            }
            println!("\n];");
        }
        "sign" => {
            let json_path = crate_root().join("catalog").join("catalog.json");
            let sig_path = crate_root().join("catalog").join("catalog.json.sig");
            let bytes = std::fs::read(&json_path).expect("read catalog.json");
            let sig = sk.sign(&bytes);
            std::fs::write(&sig_path, sig.to_bytes()).expect("write .sig");
            println!("signed {} -> {}", json_path.display(), sig_path.display());
            println!("verifying key: {}", hex::encode(vk.to_bytes()));
        }
        other => {
            eprintln!("unknown mode: {other} (use 'sign' or 'pubkey')");
            std::process::exit(2);
        }
    }
}
