//! Speech providers: turning a line into audio.
//!
//! `fish` and `openai_compatible` are HTTP services; `command` runs any local synthesizer
//! (macOS `say`, `piper`, `espeak`); `none` means the assistant works silently, which is what a
//! headless or CI run wants.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde_json::json;

use super::{check_status, with_auth, TtsProvider};

fn require(value: &str, what: &str) -> Result<String> {
    if value.trim().is_empty() {
        bail!("{what} is required by this provider but is empty in the config");
    }
    Ok(value.to_string())
}

// --------------------------------------------------------------------------- Fish Audio

pub struct FishTts {
    client: reqwest::blocking::Client,
    base_url: String,
    key: String,
    voice: String,
    model: String,
    format: String,
}

impl FishTts {
    pub fn new(
        client: reqwest::blocking::Client,
        base_url: String,
        key: String,
        voice: String,
        model: String,
        format: String,
    ) -> Self {
        Self { client, base_url: base_url.trim_end_matches('/').to_string(), key, voice, model, format }
    }
}

impl TtsProvider for FishTts {
    fn name(&self) -> String {
        format!("fish:{}", self.model)
    }

    fn format(&self) -> &str {
        &self.format
    }

    fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        // A voice that is empty would be played by the service's default voice, which is a
        // silent surprise: refuse instead.
        let voice = require(&self.voice, "tts.voice")?;
        let mut body = json!({"text": text, "format": self.format});
        body["reference_id"] = json!(voice);
        if !self.model.is_empty() {
            body["model"] = json!(self.model);
        }
        let request = self
            .client
            .post(format!("{}/tts", self.base_url))
            .header("model", &self.model)
            .json(&body);
        let response = with_auth(request, &Some(self.key.clone()))
            .send()
            .context("calling the speech service")?;
        let response = check_status(response, "the speech service")?;
        let bytes = response.bytes().context("reading the speech audio")?;
        if bytes.is_empty() {
            bail!("the speech service returned no audio");
        }
        Ok(bytes.to_vec())
    }
}

// --------------------------------------------------------------------------- OpenAI-compatible

pub struct SpeechTts {
    client: reqwest::blocking::Client,
    base_url: String,
    key: Option<String>,
    voice: String,
    model: String,
    format: String,
}

impl SpeechTts {
    pub fn new(
        client: reqwest::blocking::Client,
        base_url: String,
        key: Option<String>,
        voice: String,
        model: String,
        format: String,
    ) -> Self {
        Self { client, base_url: base_url.trim_end_matches('/').to_string(), key, voice, model, format }
    }
}

impl TtsProvider for SpeechTts {
    fn name(&self) -> String {
        format!("openai_compatible:{}", self.model)
    }

    fn format(&self) -> &str {
        &self.format
    }

    fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        let mut body = json!({"input": text, "response_format": self.format});
        if !self.voice.is_empty() {
            body["voice"] = json!(self.voice);
        }
        if !self.model.is_empty() {
            body["model"] = json!(self.model);
        }
        let request = self
            .client
            .post(format!("{}/audio/speech", self.base_url))
            .json(&body);
        let response = with_auth(request, &self.key)
            .send()
            .context("calling the speech service")?;
        let response = check_status(response, "the speech service")?;
        let bytes = response.bytes().context("reading the speech audio")?;
        if bytes.is_empty() {
            bail!("the speech service returned no audio");
        }
        Ok(bytes.to_vec())
    }
}

// --------------------------------------------------------------------------- command

/// Any local synthesizer, driven by argv.
///
/// `{file}` is replaced by a path the command must write its audio to; `{text}` by the line.
/// The line is also handed to the command on stdin, so a synthesizer that reads stdin works
/// without any placeholder at all. There is no shell involved: the argv in the config is the
/// argv that runs.
pub struct CommandTts {
    command: Vec<String>,
    format: String,
}

impl CommandTts {
    pub fn new(command: Vec<String>, format: String) -> Self {
        Self { command, format }
    }
}

impl TtsProvider for CommandTts {
    fn name(&self) -> String {
        "command".to_string()
    }

    fn format(&self) -> &str {
        &self.format
    }

    fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        if self.command.is_empty() {
            bail!("tts.provider = command needs tts.command, and it is empty");
        }
        let dir = std::env::temp_dir();
        let out_path: PathBuf = dir.join(format!(
            "hey-jev-command-{}-{}.{}",
            std::process::id(),
            // A counter would be cleaner, but a fresh name per call is enough here.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            self.format
        ));
        let argv: Vec<String> = self
            .command
            .iter()
            .map(|part| {
                part.replace("{file}", &out_path.to_string_lossy())
                    .replace("{text}", text)
            })
            .collect();

        use std::io::Write;
        let mut child = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("running the speech command `{}`", argv[0]))?;
        if let Some(stdin) = child.stdin.as_mut() {
            let _ = stdin.write_all(text.as_bytes());
        }
        let output = child.wait_with_output().context("waiting for the speech command")?;
        if !output.status.success() {
            bail!(
                "the speech command failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        if out_path.is_file() {
            let bytes = std::fs::read(&out_path).context("reading the speech command's output")?;
            let _ = std::fs::remove_file(&out_path);
            if !bytes.is_empty() {
                return Ok(bytes);
            }
        }
        // A synthesizer that wrote nothing to {file} and prints audio to stdout is supported
        // too, which is how `espeak --stdout` and friends behave.
        if output.stdout.is_empty() {
            bail!("the speech command produced no audio on stdout and wrote nothing to {{file}}");
        }
        Ok(output.stdout)
    }
}

// --------------------------------------------------------------------------- none

/// Speaks nothing. The assistant still runs its actions and still logs the line it would have
/// said, which makes the whole pipeline testable without audio hardware.
pub struct SilentTts {
    format: String,
}

impl SilentTts {
    pub fn new(format: String) -> Self {
        Self { format }
    }
}

impl TtsProvider for SilentTts {
    fn name(&self) -> String {
        "none".to_string()
    }

    fn format(&self) -> &str {
        &self.format
    }

    fn synthesize(&self, _text: &str) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
}
