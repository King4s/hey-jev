//! The controllable applications.
//!
//! Format is the original's `apps.json`, kept as-is so an existing file keeps working:
//!
//! ```json
//! {
//!   "spotify": "Spotify",
//!   "vscode": {"app": "Visual Studio Code", "say": "VS Code", "heard_as": ["this code"]}
//! }
//! ```
//!
//! The key is what the assistant answers with, `app` is the name the platform driver launches,
//! `say` is what it calls the app out loud, and `heard_as` lists what the transcriber writes
//! instead of the name — those are swapped back before the intent provider ever sees the text.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;

#[derive(Debug, Clone, Default)]
pub struct AppList {
    /// Key as the assistant says it -> the application name handed to the platform.
    pub names: BTreeMap<String, String>,
    /// Key -> the name to use when speaking about it.
    pub say: BTreeMap<String, String>,
    /// Mishearing pattern -> the name to replace it with, longest pattern first.
    pub heard_as: Vec<(Regex, String)>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Entry {
    /// The short form: `"spotify": "Spotify"`.
    Named(String),
    /// The long form, for apps the transcriber gets wrong.
    Detailed {
        app: String,
        #[serde(default)]
        say: Option<String>,
        #[serde(default)]
        heard_as: Option<Vec<String>>,
    },
}

impl AppList {
    /// Read the app list. A missing file is an error: an assistant with no apps can do nothing,
    /// and finding that out at startup beats finding it out mid-command.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading the app list {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let raw: BTreeMap<String, Entry> = serde_json::from_str(text)?;
        let mut list = AppList::default();
        let mut mishearings: Vec<(String, String)> = Vec::new();

        for (key, entry) in raw {
            match entry {
                Entry::Named(app) => {
                    list.say.insert(key.clone(), app.clone());
                    list.names.insert(key, app);
                }
                Entry::Detailed { app, say, heard_as } => {
                    let spoken = say.clone().unwrap_or_else(|| app.clone());
                    for heard in heard_as.unwrap_or_default() {
                        if !heard.trim().is_empty() {
                            mishearings.push((heard, spoken.clone()));
                        }
                    }
                    list.say.insert(key.clone(), spoken);
                    list.names.insert(key, app);
                }
            }
        }

        // Longest alternative first, so "visual studio code" is replaced before "code".
        mishearings.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        for (heard, spoken) in mishearings {
            let pattern = format!(r"\b{}\b", regex::escape(&heard));
            list.heard_as
                .push((Regex::new(&pattern).expect("escaped literal compiles"), spoken));
        }
        Ok(list)
    }

    /// The app keys, in a stable order, for building the "which app?" question's options.
    pub fn keys(&self) -> Vec<String> {
        self.names.keys().cloned().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Replace every known mishearing with the name the assistant expects.
    ///
    /// This runs before the provider sees the text, because a provider cannot pick an option
    /// that the transcriber never wrote.
    pub fn fix_names(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (pattern, replacement) in &self.heard_as {
            out = pattern.replace_all(&out, replacement.as_str()).to_string();
        }
        out
    }

    /// What to call this app out loud, falling back to the app name itself.
    pub fn spoken_name(&self, key: &str) -> Option<&str> {
        self.say.get(key).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"{
      "spotify": "Spotify",
      "vscode": {"app": "Visual Studio Code", "say": "VS Code", "heard_as": ["this code", "vs code"]}
    }"#;

    #[test]
    fn both_forms_load_and_the_longest_mishearing_wins() {
        let list = AppList::parse(EXAMPLE).unwrap();
        assert_eq!(list.keys(), vec!["spotify".to_string(), "vscode".to_string()]);
        assert_eq!(list.names.get("vscode").unwrap(), "Visual Studio Code");
        assert_eq!(list.spoken_name("vscode").unwrap(), "VS Code");
        assert_eq!(list.spoken_name("spotify").unwrap(), "Spotify");
    }

    #[test]
    fn mishearings_are_replaced_before_the_provider_sees_the_text() {
        let list = AppList::parse(EXAMPLE).unwrap();
        assert_eq!(list.fix_names("open this code"), "open VS Code");
        // A word that merely contains the mishearing must not be touched.
        assert_eq!(list.fix_names("open notice"), "open notice");
    }

    #[test]
    fn a_broken_file_is_an_error_not_an_empty_list() {
        assert!(AppList::parse("{ not json").is_err());
    }
}
