//! Platform drivers: the part that actually touches the machine.
//!
//! The macOS driver keeps the original's behaviour — it drives applications through AppleScript,
//! which is what makes "open Spotify and pause it" work on a Mac at all. The Linux driver does
//! the same jobs with `playerctl`, `pactl`, `gsettings` and friends.
//!
//! A driver is chosen at runtime from the config (`[general] platform`), so a Mac build and a
//! fleet build differ by configuration rather than by recompiling. Capabilities differ too: a
//! driver that cannot do something says so, and the assistant answers "I know what you want, I
//! just can't do that one yet" instead of failing silently.

mod linux;
mod macos;

use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, Result};

use crate::config::Config;

pub use linux::LinuxPlatform;
pub use macos::MacosPlatform;

/// Every action id a driver may implement. The reply ids in `[replies]` use the same names, and
/// `--actions` prints which of them the selected driver can actually do.
pub const ALL_ACTIONS: &[&str] = &[
    "app_open",
    "app_quit",
    "app_hide",
    "app_minimise",
    "app_focus",
    "volume_up",
    "volume_down",
    "volume_mute",
    "volume_unmute",
    "volume_set",
    "spotify_volume_up",
    "spotify_volume_down",
    "spotify_volume_mute",
    "spotify_volume_unmute",
    "spotify_volume_set",
    "display_dark_on",
    "display_dark_off",
    "display_toggle",
    "media_play",
    "media_pause",
    "media_next",
    "media_previous",
    "system_lock",
    "system_sleep",
];

pub trait Platform: Send + Sync {
    /// The driver's name, as accepted by `[general] platform`.
    fn name(&self) -> &'static str;

    /// Run one action. `arg` is the app key, the volume level or whatever the target's
    /// argument question produced.
    fn run(&self, action: &str, arg: Option<&str>, config: &Config) -> Result<()>;

    /// Play a rendered audio file.
    fn play_audio(&self, file: &Path) -> Result<()>;

    /// Put text on the clipboard.
    fn copy_to_clipboard(&self, text: &str) -> Result<()>;

    /// Open a URL, optionally in a named browser.
    fn open_url(&self, browser: Option<&str>, url: &str, config: &Config) -> Result<()>;

    /// Open a new tab in a browser.
    fn new_tab(&self, browser: Option<&str>, config: &Config) -> Result<()>;

    /// Which of the known apps is in front, if any. Used to answer "new tab" with the browser
    /// the user is already looking at.
    fn front_app(&self, config: &Config) -> Result<Option<String>>;

    /// Whether this driver implements an action at all.
    fn supports(&self, action: &str) -> bool;
}

/// Build the driver named in `[general] platform`, or the one for this host when it is unset.
pub fn build(configured: Option<&str>, config: &Config) -> Result<Box<dyn Platform>> {
    let _ = config;
    match configured.unwrap_or("") {
        "" | "auto" => Ok(default_for_host()),
        "macos" => Ok(Box::new(MacosPlatform::new())),
        "linux" => Ok(Box::new(LinuxPlatform::new())),
        "recording" => Ok(Box::new(RecordingPlatform::default())),
        other => bail!("unknown platform `{other}` — known drivers: macos, linux, recording, auto"),
    }
}

/// The driver for the host this binary was built for.
pub fn default_for_host() -> Box<dyn Platform> {
    #[cfg(target_os = "macos")]
    {
        Box::new(MacosPlatform::new())
    }
    #[cfg(target_os = "linux")]
    {
        Box::new(LinuxPlatform::new())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Box::new(RecordingPlatform::default())
    }
}

/// Resolve an app key to the name the driver launches.
pub(crate) fn app_name<'a>(config: &'a Config, key: &str) -> Result<&'a str> {
    config
        .apps_list
        .names
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| anyhow::anyhow!("`{key}` is not in the app list ({})", config.apps_file().display()))
}

/// A driver that records what it was asked to do and does nothing else.
///
/// This is what `--dry-run` uses: it proves the whole pipeline — provider, decision, action
/// selection — end to end on a machine that has none of the applications involved.
#[derive(Default)]
pub struct RecordingPlatform {
    calls: Mutex<Vec<String>>,
}

impl RecordingPlatform {
    /// Everything this driver was asked to do, in order.
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("recording lock").clone()
    }

    fn record(&self, call: String) -> Result<()> {
        self.calls.lock().expect("recording lock").push(call);
        Ok(())
    }
}

impl Platform for RecordingPlatform {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn run(&self, action: &str, arg: Option<&str>, _config: &Config) -> Result<()> {
        self.record(match arg {
            Some(arg) => format!("{action} {arg}"),
            None => action.to_string(),
        })
    }

    fn play_audio(&self, file: &Path) -> Result<()> {
        self.record(format!("play {}", file.display()))
    }

    fn copy_to_clipboard(&self, text: &str) -> Result<()> {
        self.record(format!("clipboard {} chars", text.chars().count()))
    }

    fn open_url(&self, browser: Option<&str>, url: &str, _config: &Config) -> Result<()> {
        self.record(match browser {
            Some(browser) => format!("open {url} in {browser}"),
            None => format!("open {url}"),
        })
    }

    fn new_tab(&self, browser: Option<&str>, _config: &Config) -> Result<()> {
        self.record(match browser {
            Some(browser) => format!("new tab in {browser}"),
            None => "new tab".to_string(),
        })
    }

    fn front_app(&self, _config: &Config) -> Result<Option<String>> {
        Ok(None)
    }

    fn supports(&self, action: &str) -> bool {
        ALL_ACTIONS.contains(&action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn config() -> Config {
        let mut config = Config::from_toml(crate::config::EXAMPLE_CONFIG).unwrap();
        config.apps_list.names.insert("spotify".into(), "Spotify".into());
        config
    }

    #[test]
    fn the_recording_driver_records_instead_of_acting() {
        let platform = RecordingPlatform::default();
        let config = config();
        platform.run("app_open", Some("spotify"), &config).unwrap();
        platform.open_url(Some("chrome"), "https://example.com", &config).unwrap();
        assert_eq!(
            platform.calls(),
            vec!["app_open spotify".to_string(), "open https://example.com in chrome".to_string()]
        );
    }

    #[test]
    fn an_unknown_platform_is_an_error_that_lists_the_known_ones() {
        let config = config();
        match build(Some("windows"), &config) {
            Ok(platform) => panic!("`windows` should not resolve, got {}", platform.name()),
            Err(error) => {
                let error = error.to_string();
                assert!(error.contains("macos") && error.contains("linux"));
            }
        }
        assert_eq!(build(Some("recording"), &config).unwrap().name(), "recording");
    }

    #[test]
    fn an_unknown_app_says_which_file_to_edit() {
        let config = config();
        let error = app_name(&config, "nope").unwrap_err().to_string();
        assert!(error.contains("apps.json"));
    }
}
