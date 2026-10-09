//! The macOS driver: applications driven through AppleScript.
//!
//! This is a direct port of the original's action table, including the two behaviours that look
//! like bugs and are not: `media_play` keeps asking, because Spotify ignores play while it is
//! still loading, and `media_previous` presses twice unless the song has just started, because
//! one press only restarts the current track.

use std::path::Path;
use std::process::Command as Proc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::config::Config;

use super::{app_name, Platform, ALL_ACTIONS};

pub struct MacosPlatform;

impl MacosPlatform {
    pub fn new() -> Self {
        Self
    }

    /// Run an AppleScript, returning its stdout.
    fn osa(&self, script: &str) -> Result<String> {
        let output = Proc::new("osascript")
            .arg("-e")
            .arg(script)
            .output()
            .with_context(|| format!("running osascript: {script}"))?;
        if !output.status.success() {
            bail!("AppleScript failed: {}", String::from_utf8_lossy(&output.stderr).trim());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Run a program directly.
    fn sh(&self, argv: &[String]) -> Result<()> {
        let (program, rest) = argv.split_first().context("empty command")?;
        let output = Proc::new(program).args(rest).output().with_context(|| format!("running {program}"))?;
        if !output.status.success() {
            bail!(
                "{program} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(())
    }

    fn output_volume(&self) -> Result<u8> {
        Ok(self.osa("output volume of (get volume settings)")?.parse().unwrap_or(50))
    }

    fn spotify_volume(&self) -> Result<u8> {
        Ok(self
            .osa("tell application \"Spotify\" to get sound volume")?
            .parse()
            .unwrap_or(50))
    }

    /// Launch an app and wait until it reports running, so a follow-up action does not land
    /// before the app exists.
    fn open_app(&self, name: &str) -> Result<()> {
        self.sh(&["open".into(), "-a".into(), name.to_string()])?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if self.osa(&format!("application \"{name}\" is running"))? == "true" {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        Ok(())
    }

    /// Spotify ignores play while it is still loading, so keep asking until it says it is playing.
    fn spotify_play(&self) -> Result<()> {
        for _ in 0..12 {
            self.osa("tell application \"Spotify\" to play")?;
            std::thread::sleep(Duration::from_millis(500));
            if self.osa("tell application \"Spotify\" to player state")? == "playing" {
                return Ok(());
            }
        }
        bail!("Spotify never started playing")
    }

    /// A System Events handle for a running app, found by bundle id because process names differ
    /// from application names (VS Code runs as "Code").
    fn app_process(&self, name: &str) -> Result<String> {
        let bundle = self.osa(&format!("id of application \"{name}\""))?;
        Ok(format!("(first application process whose bundle identifier is \"{bundle}\")"))
    }
}

impl Default for MacosPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl Platform for MacosPlatform {
    fn name(&self) -> &'static str {
        "macos"
    }

    fn run(&self, action: &str, arg: Option<&str>, config: &Config) -> Result<()> {
        let ceiling = config.general.volume.ceiling();
        let step = config.general.volume.step;
        match action {
            "app_open" => self.open_app(app_name(config, required(arg, action)?)?),
            "app_quit" => {
                self.osa(&format!("tell application \"{}\" to quit", app_name(config, required(arg, action)?)?))?;
                Ok(())
            }
            "app_focus" => {
                self.osa(&format!("tell application \"{}\" to activate", app_name(config, required(arg, action)?)?))?;
                Ok(())
            }
            "app_hide" => {
                let process = self.app_process(app_name(config, required(arg, action)?)?)?;
                self.osa(&format!(
                    "tell application \"System Events\" to set visible of {process} to false"
                ))?;
                Ok(())
            }
            "app_minimise" => {
                let process = self.app_process(app_name(config, required(arg, action)?)?)?;
                self.osa(&format!(
                    "tell application \"System Events\" to set value of attribute \"AXMinimized\" of every window of {process} to true"
                ))?;
                Ok(())
            }
            "volume_up" => {
                let level = (self.output_volume()? as u16 + step as u16).min(ceiling as u16);
                self.osa(&format!("set volume output volume {level}"))?;
                Ok(())
            }
            "volume_down" => {
                let level = (self.output_volume()? as u16).saturating_sub(step as u16);
                self.osa(&format!("set volume output volume {level}"))?;
                Ok(())
            }
            "volume_mute" => {
                self.osa("set volume output muted true")?;
                Ok(())
            }
            "volume_unmute" => {
                self.osa("set volume output muted false")?;
                Ok(())
            }
            "volume_set" => {
                let level = config.general.volume.level(required(arg, action)?)?;
                self.osa(&format!("set volume output volume {level}"))?;
                Ok(())
            }
            "spotify_volume_up" => {
                let level = (self.spotify_volume()? as u16 + step as u16).min(ceiling as u16);
                self.osa(&format!("tell application \"Spotify\" to set sound volume to {level}"))?;
                Ok(())
            }
            "spotify_volume_down" => {
                let level = (self.spotify_volume()? as u16).saturating_sub(step as u16);
                self.osa(&format!("tell application \"Spotify\" to set sound volume to {level}"))?;
                Ok(())
            }
            "spotify_volume_mute" => {
                self.osa("tell application \"Spotify\" to set sound volume to 0")?;
                Ok(())
            }
            "spotify_volume_unmute" => {
                self.osa(&format!(
                    "tell application \"Spotify\" to set sound volume to {}",
                    config.general.volume.ceiling().min(50)
                ))?;
                Ok(())
            }
            "spotify_volume_set" => {
                let level = config.general.volume.level(required(arg, action)?)?;
                self.osa(&format!("tell application \"Spotify\" to set sound volume to {level}"))?;
                Ok(())
            }
            "display_dark_on" => {
                self.osa("tell application \"System Events\" to tell appearance preferences to set dark mode to true")?;
                Ok(())
            }
            "display_dark_off" => {
                self.osa("tell application \"System Events\" to tell appearance preferences to set dark mode to false")?;
                Ok(())
            }
            "display_toggle" => {
                self.osa("tell application \"System Events\" to tell appearance preferences to set dark mode to not dark mode")?;
                Ok(())
            }
            "media_play" => self.spotify_play(),
            "media_pause" => {
                self.osa("tell application \"Spotify\" to pause")?;
                Ok(())
            }
            "media_next" => {
                self.osa("tell application \"Spotify\" to next track")?;
                Ok(())
            }
            // Mid-song, one "previous" only restarts it, so press twice unless we are in the
            // first three seconds.
            "media_previous" => {
                self.osa(
                    "tell application \"Spotify\"\nif player position > 3 then\nprevious track\ndelay 0.3\nend if\nprevious track\nend tell",
                )?;
                Ok(())
            }
            "system_lock" => {
                self.osa("tell application \"System Events\" to keystroke \"q\" using {control down, command down}")?;
                Ok(())
            }
            "system_sleep" => self.sh(&["pmset".into(), "sleepnow".into()]),
            other => bail!("the macOS driver does not implement `{other}`"),
        }
    }

    fn play_audio(&self, file: &Path) -> Result<()> {
        self.sh(&["afplay".into(), file.to_string_lossy().to_string()])
    }

    fn copy_to_clipboard(&self, text: &str) -> Result<()> {
        use std::io::Write;
        let mut child = Proc::new("pbcopy")
            .stdin(std::process::Stdio::piped())
            .spawn()
            .context("running pbcopy")?;
        child
            .stdin
            .as_mut()
            .context("pbcopy took no stdin")?
            .write_all(text.as_bytes())
            .context("writing to pbcopy")?;
        let status = child.wait().context("waiting for pbcopy")?;
        if !status.success() {
            bail!("pbcopy failed");
        }
        Ok(())
    }

    fn open_url(&self, browser: Option<&str>, url: &str, config: &Config) -> Result<()> {
        match browser {
            Some(key) => self.sh(&[
                "open".into(),
                "-a".into(),
                app_name(config, key)?.to_string(),
                url.to_string(),
            ]),
            None => self.sh(&["open".into(), url.to_string()]),
        }
    }

    fn new_tab(&self, browser: Option<&str>, config: &Config) -> Result<()> {
        let name = match browser {
            Some(key) => app_name(config, key)?.to_string(),
            None => bail!("no browser named and none in front"),
        };
        self.sh(&["open".into(), "-a".into(), name.clone()])?;
        self.osa(&format!("tell application \"{name}\" to activate"))?;
        std::thread::sleep(Duration::from_millis(300));
        self.osa("tell application \"System Events\" to keystroke \"t\" using command down")?;
        Ok(())
    }

    fn front_app(&self, config: &Config) -> Result<Option<String>> {
        let front = self
            .osa("tell application \"System Events\" to get name of first application process whose frontmost is true")?;
        Ok(config
            .apps_list
            .names
            .iter()
            .find(|(_, name)| name.as_str() == front)
            .map(|(key, _)| key.clone()))
    }

    fn supports(&self, action: &str) -> bool {
        ALL_ACTIONS.contains(&action)
    }
}

/// The argument an action cannot run without.
pub fn required<'a>(arg: Option<&'a str>, action: &str) -> Result<&'a str> {
    arg.filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("`{action}` needs an argument and none was resolved"))
}
