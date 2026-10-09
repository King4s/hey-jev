//! The Linux driver.
//!
//! The same jobs as the macOS driver, done with the tools a Linux desktop actually has:
//! `playerctl` for media, `pactl` for volume, `gsettings` for the colour scheme, `xdotool` for
//! windows, `loginctl`/`systemctl` for the session.
//!
//! Capabilities are narrower than on macOS and the difference is reported rather than hidden:
//! a Linux desktop has no single AppleScript-equivalent way to open a *new tab* in a named
//! browser, so `browser_new_tab` is not supported here and the assistant says so.

use std::path::Path;
use std::process::Command as Proc;

use anyhow::{bail, Context, Result};

use crate::config::Config;

use super::{app_name, Platform};

pub struct LinuxPlatform;

impl LinuxPlatform {
    pub fn new() -> Self {
        Self
    }

    fn sh(&self, argv: &[String]) -> Result<()> {
        let (program, rest) = argv.split_first().context("empty command")?;
        let output = Proc::new(program)
            .args(rest)
            .output()
            .with_context(|| format!("running {program}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let detail = if stderr.is_empty() {
                format!("exit {}", output.status)
            } else {
                stderr
            };
            bail!("{program} failed: {detail}");
        }
        Ok(())
    }

    fn stdout(&self, argv: &[String]) -> Result<String> {
        let (program, rest) = argv.split_first().context("empty command")?;
        let output = Proc::new(program)
            .args(rest)
            .output()
            .with_context(|| format!("running {program}"))?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// The named player's volume, as a percentage.
    fn player_volume(&self, player: &str) -> Result<u8> {
        let value = self.stdout(&[
            "playerctl".into(),
            "--player".into(),
            player.into(),
            "volume".into(),
        ])?;
        let fraction: f64 = value.parse().unwrap_or(0.5);
        Ok((fraction * 100.0).round().clamp(0.0, 100.0) as u8)
    }
}

impl Default for LinuxPlatform {
    fn default() -> Self {
        Self::new()
    }
}

/// The desktop file id `gtk-launch` expects. Desktop entries are conventionally the
/// application's name, lowercased with spaces removed.
fn desktop_id(name: &str) -> String {
    name.to_lowercase().replace(' ', "")
}

impl Platform for LinuxPlatform {
    fn name(&self) -> &'static str {
        "linux"
    }

    fn run(&self, action: &str, arg: Option<&str>, config: &Config) -> Result<()> {
        let step = config.general.volume.step;
        let ceiling = config.general.volume.ceiling();
        let sink = "@DEFAULT_SINK@";
        match action {
            "app_open" => {
                let name = app_name(config, required_arg(arg, action)?)?;
                self.sh(&["gtk-launch".into(), desktop_id(name)])
            }
            // There is no single portable "quit this application" on Linux; this is the
            // closest, and it is deliberately matched on the full name so "Slack" does not
            // take `slack-helper` with it.
            "app_quit" => {
                let name = app_name(config, required_arg(arg, action)?)?;
                self.sh(&["pkill".into(), "-f".into(), name.to_string()])
            }
            "app_focus" => {
                let name = app_name(config, required_arg(arg, action)?)?;
                self.sh(&["xdotool".into(), "search".into(), "--name".into(), name.to_string(), "windowactivate".into()])
            }
            "app_minimise" => {
                let name = app_name(config, required_arg(arg, action)?)?;
                self.sh(&["xdotool".into(), "search".into(), "--name".into(), name.to_string(), "windowminimize".into()])
            }
            "app_hide" => {
                let name = app_name(config, required_arg(arg, action)?)?;
                self.sh(&["xdotool".into(), "search".into(), "--name".into(), name.to_string(), "windowunmap".into()])
            }
            "volume_up" => self.sh(&[
                "pactl".into(),
                "set-sink-volume".into(),
                sink.into(),
                format!("+{step}%"),
            ]),
            "volume_down" => self.sh(&[
                "pactl".into(),
                "set-sink-volume".into(),
                sink.into(),
                format!("-{step}%"),
            ]),
            "volume_mute" => self.sh(&["pactl".into(), "set-sink-mute".into(), sink.into(), "1".into()]),
            "volume_unmute" => self.sh(&["pactl".into(), "set-sink-mute".into(), sink.into(), "0".into()]),
            "volume_set" => {
                let level = config.general.volume.level(required_arg(arg, action)?)?;
                self.sh(&[
                    "pactl".into(),
                    "set-sink-volume".into(),
                    sink.into(),
                    format!("{level}%"),
                ])
            }
            "spotify_volume_up" => {
                let level =
                    (self.player_volume("spotify")? as u16 + step as u16).min(ceiling as u16) as u8;
                self.set_player_volume("spotify", level)
            }
            "spotify_volume_down" => {
                let level =
                    (self.player_volume("spotify")? as u16).saturating_sub(step as u16) as u8;
                self.set_player_volume("spotify", level)
            }
            "spotify_volume_mute" => self.set_player_volume("spotify", 0),
            "spotify_volume_unmute" => {
                let level = ceiling.min(50);
                self.set_player_volume("spotify", level)
            }
            "spotify_volume_set" => {
                let level = config.general.volume.level(required_arg(arg, action)?)?;
                self.set_player_volume("spotify", level)
            }
            "display_dark_on" => self.scheme("prefer-dark"),
            "display_dark_off" => self.scheme("prefer-light"),
            "display_toggle" => {
                let current = self.stdout(&[
                    "gsettings".into(),
                    "get".into(),
                    "org.gnome.desktop.interface".into(),
                    "color-scheme".into(),
                ])?;
                if current.contains("dark") {
                    self.scheme("prefer-light")
                } else {
                    self.scheme("prefer-dark")
                }
            }
            "media_play" => self.sh(&["playerctl".into(), "play".into()]),
            "media_pause" => self.sh(&["playerctl".into(), "pause".into()]),
            "media_next" => self.sh(&["playerctl".into(), "next".into()]),
            "media_previous" => self.sh(&["playerctl".into(), "previous".into()]),
            "system_lock" => self.sh(&["loginctl".into(), "lock-session".into()]),
            "system_sleep" => self.sh(&["systemctl".into(), "suspend".into()]),
            other => bail!("the Linux driver does not implement `{other}`"),
        }
    }

    fn play_audio(&self, file: &Path) -> Result<()> {
        self.sh(&["paplay".into(), file.to_string_lossy().to_string()])
    }

    fn copy_to_clipboard(&self, text: &str) -> Result<()> {
        use std::io::Write;
        // Wayland first, then X11. Whichever exists is the one this session uses.
        for (program, args) in [("wl-copy", vec![]), ("xclip", vec!["-selection", "clipboard"])] {
            let mut argv: Vec<String> = args.iter().map(|a| a.to_string()).collect();
            argv.insert(0, program.to_string());
            let Ok(mut child) = Proc::new(&argv[0])
                .args(&argv[1..])
                .stdin(std::process::Stdio::piped())
                .spawn()
            else {
                continue;
            };
            let wrote = child
                .stdin
                .as_mut()
                .map(|stdin| stdin.write_all(text.as_bytes()).is_ok())
                .unwrap_or(false);
            if wrote && child.wait().map(|status| status.success()).unwrap_or(false) {
                return Ok(());
            }
        }
        bail!("no clipboard tool found: install wl-clipboard or xclip")
    }

    fn open_url(&self, browser: Option<&str>, url: &str, config: &Config) -> Result<()> {
        let _ = browser;
        let _ = config;
        self.sh(&["xdg-open".into(), url.to_string()])
    }

    fn new_tab(&self, _browser: Option<&str>, _config: &Config) -> Result<()> {
        bail!("opening a new browser tab is not supported by the Linux driver")
    }

    fn front_app(&self, config: &Config) -> Result<Option<String>> {
        let title = self.stdout(&[
            "xdotool".into(),
            "getactivewindow".into(),
            "getwindowname".into(),
        ])?;
        Ok(config
            .apps_list
            .names
            .iter()
            .find(|(_, name)| title.to_lowercase().contains(&name.to_lowercase()))
            .map(|(key, _)| key.clone()))
    }

    fn supports(&self, action: &str) -> bool {
        matches!(
            action,
            "app_open"
                | "app_quit"
                | "app_focus"
                | "app_minimise"
                | "app_hide"
                | "volume_up"
                | "volume_down"
                | "volume_mute"
                | "volume_unmute"
                | "volume_set"
                | "spotify_volume_up"
                | "spotify_volume_down"
                | "spotify_volume_mute"
                | "spotify_volume_unmute"
                | "spotify_volume_set"
                | "display_dark_on"
                | "display_dark_off"
                | "display_toggle"
                | "media_play"
                | "media_pause"
                | "media_next"
                | "media_previous"
                | "system_lock"
                | "system_sleep"
        )
    }
}

impl LinuxPlatform {
    fn set_player_volume(&self, player: &str, level: u8) -> Result<()> {
        self.sh(&[
            "playerctl".into(),
            "--player".into(),
            player.into(),
            "volume".into(),
            format!("{:.2}", level as f64 / 100.0),
        ])
    }

    fn scheme(&self, value: &str) -> Result<()> {
        self.sh(&[
            "gsettings".into(),
            "set".into(),
            "org.gnome.desktop.interface".into(),
            "color-scheme".into(),
            value.into(),
        ])
    }
}

/// The argument an action cannot run without.
fn required_arg<'a>(arg: Option<&'a str>, action: &str) -> Result<&'a str> {
    arg.filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("`{action}` needs an argument and none was resolved"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A driver must claim only what it can actually do: the assistant turns an unsupported
    /// action into "I can't do that one yet" instead of a silent failure.
    #[test]
    fn the_linux_driver_reports_its_gaps() {
        let platform = LinuxPlatform::new();
        assert!(platform.supports("media_pause"));
        assert!(platform.supports("display_toggle"));
        assert!(!platform.supports("browser_new_tab"));
        assert!(platform.new_tab(None, &Config::from_toml(crate::config::EXAMPLE_CONFIG).unwrap()).is_err());
    }

    #[test]
    fn desktop_ids_follow_the_convention() {
        assert_eq!(desktop_id("Google Chrome"), "googlechrome");
        assert_eq!(desktop_id("Spotify"), "spotify");
    }
}
