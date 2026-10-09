//! Speaking: render a line once, cache the audio, play it.
//!
//! The original pre-rendered every scripted reply on first launch so that answers were instant
//! and cost nothing after the first run. Same idea here: a line is rendered on demand and kept
//! on disk under a hash of the provider, the voice and the text, so changing the voice does not
//! serve stale audio and changing nothing serves the same file.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha1::{Digest, Sha1};

use crate::providers::TtsProvider;
use crate::platform::Platform;

pub struct Speaker {
    provider: Box<dyn TtsProvider>,
    cache_dir: PathBuf,
}

impl Speaker {
    pub fn new(provider: Box<dyn TtsProvider>, cache_dir: PathBuf) -> Self {
        Self { provider, cache_dir }
    }

    pub fn provider_name(&self) -> String {
        self.provider.name()
    }

    /// The cache file for a line, whether or not it exists yet.
    pub fn cache_path(&self, line: &str) -> PathBuf {
        let mut hasher = Sha1::new();
        hasher.update(self.provider.name().as_bytes());
        hasher.update(b"|");
        hasher.update(line.as_bytes());
        let digest = hasher.finalize();
        self.cache_dir.join(format!("{:x}.{}", digest, self.provider.format()))
    }

    /// Render a line to a file, reusing the cache. `None` when the provider is silent.
    ///
    /// Returns `(path, was_cached)`.
    pub fn render(&self, line: &str) -> Result<Option<(PathBuf, bool)>> {
        let path = self.cache_path(line);
        if path.is_file() {
            return Ok(Some((path, true)));
        }
        let audio = self.provider.synthesize(line)?;
        if audio.is_empty() {
            return Ok(None);
        }
        std::fs::create_dir_all(&self.cache_dir)
            .with_context(|| format!("creating the audio cache {}", self.cache_dir.display()))?;
        // Write to a temporary name and rename, so a half-written file is never played or
        // mistaken for a cached one.
        let temporary = path.with_extension("part");
        std::fs::write(&temporary, &audio)
            .with_context(|| format!("writing {}", temporary.display()))?;
        std::fs::rename(&temporary, &path)
            .with_context(|| format!("moving {} into place", temporary.display()))?;
        Ok(Some((path, false)))
    }

    /// Render and play a line. Returns the file that was played, if any.
    pub fn say(&self, line: &str, platform: &dyn Platform) -> Result<Option<PathBuf>> {
        match self.render(line)? {
            Some((path, _)) => {
                platform.play_audio(&path)?;
                Ok(Some(path))
            }
            None => Ok(None),
        }
    }

    /// Render every line the config can produce, so runtime replies are instant.
    ///
    /// Only lines with no live placeholder are pre-renderable; `{left}` and `{label}` depend on
    /// the moment and are rendered when needed.
    pub fn warm(&self, lines: &[String]) -> Result<usize> {
        let mut rendered = 0;
        for line in lines {
            if line.contains('{') {
                continue;
            }
            match self.render(line) {
                Ok(Some((_, false))) => rendered += 1,
                Ok(_) => {}
                Err(error) => eprintln!("  could not pre-render {line:?}: {error}"),
            }
        }
        Ok(rendered)
    }
}

/// Every line the config can produce for a reply key, with `{app}` and `{level}` expanded.
pub fn expand_lines(
    replies: &crate::config::Replies,
    app_names: &[&str],
    level_names: &[&str],
) -> Vec<String> {
    let mut out = Vec::new();
    for lines in replies.lines.values() {
        for line in lines {
            if line.contains("{app}") {
                for app in app_names {
                    out.push(line.replace("{app}", app));
                }
            } else if line.contains("{level}") {
                for level in level_names {
                    out.push(line.replace("{level}", level));
                }
            } else {
                out.push(line.clone());
            }
        }
    }
    out
}

/// Play a sound file that is not speech (the wake chime).
pub fn play_file(platform: &dyn Platform, path: &Path) -> Result<()> {
    platform.play_audio(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::tts::SilentTts;

    fn speaker() -> Speaker {
        Speaker::new(Box::new(SilentTts::new("wav".into())), std::env::temp_dir().join("hey-jev-test-cache"))
    }

    /// The same line must always land on the same file, and a different line must not.
    #[test]
    fn the_cache_key_follows_the_text() {
        let speaker = speaker();
        assert_eq!(speaker.cache_path("hello"), speaker.cache_path("hello"));
        assert_ne!(speaker.cache_path("hello"), speaker.cache_path("hello!"));
    }

    /// A silent provider produces no file at all, so nothing is played and nothing is cached.
    #[test]
    fn a_silent_provider_renders_nothing() {
        assert!(speaker().render("hello").unwrap().is_none());
    }

    #[test]
    fn lines_with_placeholders_are_expanded_per_app_and_level() {
        let replies = crate::config::Replies {
            lines: std::collections::BTreeMap::from([
                ("app_open".to_string(), vec!["There you go.".to_string()]),
                ("app_quit".to_string(), vec!["{app}'s gone.".to_string()]),
            ]),
        };
        let lines = expand_lines(&replies, &["Spotify", "Slack"], &[]);
        assert!(lines.contains(&"There you go.".to_string()));
        assert!(lines.contains(&"Spotify's gone.".to_string()));
        assert!(lines.contains(&"Slack's gone.".to_string()));
    }

    /// A live placeholder cannot be pre-rendered, so warming must skip it rather than render a
    /// line that says the literal "{left}".
    #[test]
    fn warming_skips_lines_that_depend_on_the_moment() {
        let speaker = speaker();
        let lines = vec!["{left} left.".to_string()];
        assert_eq!(speaker.warm(&lines).unwrap(), 0);
    }
}
