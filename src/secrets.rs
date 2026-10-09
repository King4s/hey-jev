//! Key resolution.
//!
//! A key is never stored in the config file; the config names it (`key_env = "FISH_AUDIO_API_KEY"`)
//! and this module finds its value. Order, first hit wins:
//!
//! 1. the process environment,
//! 2. any `.env` file listed in `[keys] dotenv_files`, in order,
//! 3. the macOS login keychain, under `[keys] keychain_service`.
//!
//! The environment deliberately wins over the keychain so that editing a `.env` always takes
//! effect, which is the behaviour the original had and what makes "it still says 401" debuggable.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{bail, Result};

#[cfg(target_os = "macos")]
use std::process::Command;

use crate::config::Config;

#[derive(Debug, Clone)]
pub struct Secrets {
    /// Values read from `.env` files, in the order the files were listed.
    dotenv: BTreeMap<String, String>,
    keychain_service: String,
    /// Cached keychain lookups, so a burst of TTS calls does not shell out repeatedly.
    cache: BTreeMap<String, Option<String>>,
}

impl Secrets {
    /// Build from config, reading every configured `.env` file that exists.
    pub fn from_config(config: &Config) -> Self {
        let mut dotenv = BTreeMap::new();
        for file in &config.keys.dotenv_files {
            let path = config.resolve(file);
            match std::fs::read_to_string(&path) {
                Ok(text) => dotenv.extend(parse_dotenv(&text)),
                Err(err) => eprintln!("  ignoring unreadable dotenv {}: {err}", path.display()),
            }
        }
        Self {
            dotenv,
            keychain_service: config.keys.keychain_service.clone(),
            cache: BTreeMap::new(),
        }
    }

    /// Look up a key by name. `None` when it is set nowhere.
    pub fn get(&mut self, name: &str) -> Option<String> {
        if name.is_empty() {
            return None;
        }
        if let Ok(value) = std::env::var(name) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
        if let Some(value) = self.dotenv.get(name) {
            if !value.is_empty() {
                return Some(value.clone());
            }
        }
        let cached = self
            .cache
            .entry(name.to_string())
            .or_insert_with(|| keychain_value(&self.keychain_service, name));
        cached.clone()
    }

    /// Look up a key that the chosen provider cannot work without.
    pub fn require(&mut self, name: &str) -> Result<String> {
        match self.get(name) {
            Some(value) => Ok(value),
            None => bail!(
                "no value for {name}: put it in the environment, in a .env file, \
                 or in the keychain entry `{name}` of service `{}`",
                self.keychain_service
            ),
        }
    }
}

/// Read a `.env`-style file: `NAME=value`, `#` comments, optional `export`, optional quotes.
///
/// Deliberately minimal — it is a key file, not a shell. A value is taken literally so a key
/// containing `#` or `$` survives.
pub fn parse_dotenv(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let value = value.trim();
        let value = match value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
            Some(inner) => inner,
            None => match value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')) {
                Some(inner) => inner,
                None => value,
            },
        };
        out.insert(name.to_string(), value.to_string());
    }
    out
}

#[cfg(target_os = "macos")]
fn keychain_value(service: &str, name: &str) -> Option<String> {
    let output = Command::new("security")
        .args(["find-generic-password", "-a", name, "-s", service, "-w"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!value.is_empty()).then_some(value)
}

/// Store a key in the macOS keychain, replacing any existing entry.
#[cfg(target_os = "macos")]
pub fn keychain_save(service: &str, name: &str, value: &str) -> Result<()> {
    let value = value.trim();
    if value.is_empty() {
        bail!("refusing to store an empty value for {name}");
    }
    let status = Command::new("security")
        .args(["add-generic-password", "-U", "-a", name, "-s", service, "-w", value])
        .status()?;
    if !status.success() {
        bail!("`security add-generic-password` failed for {name}");
    }
    Ok(())
}

/// Other platforms have no keychain here; environment and `.env` are the whole story.
#[cfg(not(target_os = "macos"))]
fn keychain_value(_service: &str, _name: &str) -> Option<String> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn keychain_save(_service: &str, _name: &str, _value: &str) -> Result<()> {
    bail!("this platform has no keychain backend; use the environment or a .env file")
}

/// The dotenv path beside the config, which is where a user would naturally put one.
pub fn default_dotenv_path(config: &Config) -> Option<PathBuf> {
    config.base_dir().map(|dir| dir.join(".env"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotenv_parsing_handles_the_awkward_cases() {
        let parsed = parse_dotenv(
            "# a comment\n\
             export FOO=bar\n\
             QUOTED=\"a b\"\n\
             SINGLE='c d'\n\
             WITH_HASH=abc#def\n\
             EMPTY=\n\
             \n\
             not a pair\n",
        );
        assert_eq!(parsed.get("FOO").unwrap(), "bar");
        assert_eq!(parsed.get("QUOTED").unwrap(), "a b");
        assert_eq!(parsed.get("SINGLE").unwrap(), "c d");
        // A key containing # must survive literally: these keys are opaque.
        assert_eq!(parsed.get("WITH_HASH").unwrap(), "abc#def");
        assert_eq!(parsed.get("EMPTY").unwrap(), "");
        assert!(!parsed.contains_key("not a pair"));
    }

    /// The environment must win, otherwise editing a .env appears to do nothing.
    #[test]
    fn environment_beats_dotenv() {
        let mut secrets = Secrets {
            dotenv: BTreeMap::from([("HEY_JEV_TEST_KEY".to_string(), "from-file".to_string())]),
            keychain_service: "none".into(),
            cache: BTreeMap::new(),
        };
        assert_eq!(secrets.get("HEY_JEV_TEST_KEY").unwrap(), "from-file");
        std::env::set_var("HEY_JEV_TEST_KEY", "from-env");
        assert_eq!(secrets.get("HEY_JEV_TEST_KEY").unwrap(), "from-env");
        std::env::remove_var("HEY_JEV_TEST_KEY");
        assert!(secrets.require("HEY_JEV_DEFINITELY_MISSING").is_err());
        assert!(secrets.get("").is_none());
    }
}
