//! catsign — sign the bundled catalog with the project's ed25519 signing key (CAT-5/CAT-6).
//!
//! The signing key is derived deterministically from a fixed development seed so the
//! verifying key is stable and can be baked into the app binary (`catalog::BAKED_PUBKEY`).
//! In a real release the seed would come from a secret store / HSM, not source — this
//! keeps the *mechanism* honest and testable end to end while remaining reproducible.
//!
//! Usage:
//!   cargo run --bin catsign -- sign      # writes catalog/catalog.json.sig
//!   cargo run --bin catsign -- pubkey    # prints the [u8;32] verifying key literal

use ed25519_dalek::{Signer, SigningKey};
use std::path::PathBuf;

// Fixed development seed. NOT a production secret.
const DEV_SEED: [u8; 32] = *b"kayon-catalog-dev-signing-seed01";

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn signing_key() -> SigningKey {
    SigningKey::from_bytes(&DEV_SEED)
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
