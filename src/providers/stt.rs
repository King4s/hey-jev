//! Transcription providers: turning recorded audio into text.
//!
//! `command` is the local option (whisper.cpp and anything else that takes a file and prints a
//! transcript); `openai_compatible` is the cloud option, which is the one that sends audio off
//! the machine. That distinction belongs in the config, where it can be read and changed.

use anyhow::{bail, Context, Result};

use super::{check_status, with_auth, SttProvider};

/// The audio file handed to a command provider is written with this name.
pub const AUDIO_FILENAME: &str = "audio.wav";

pub struct OpenAiCompatibleStt {
    client: reqwest::blocking::Client,
    base_url: String,
    key: Option<String>,
    model: String,
    language: String,
    prompt: Option<String>,
}

impl OpenAiCompatibleStt {
    pub fn new(
        client: reqwest::blocking::Client,
        base_url: String,
        key: Option<String>,
        model: String,
        language: String,
        prompt: Option<String>,
    ) -> Self {
        Self { client, base_url: base_url.trim_end_matches('/').to_string(), key, model, language, prompt }
    }
}

impl SttProvider for OpenAiCompatibleStt {
    fn name(&self) -> String {
        format!("openai_compatible:{}", self.model)
    }

    fn transcribe(&self, audio: &[u8], filename: &str) -> Result<String> {
        if self.model.trim().is_empty() {
            bail!("stt.model is required by this provider but is empty in the config");
        }
        let mut form = reqwest::blocking::multipart::Form::new()
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(audio.to_vec())
                    .file_name(filename.to_string())
                    .mime_str("audio/wav")?,
            )
            .text("model", self.model.clone());
        if !self.language.is_empty() {
            form = form.text("language", self.language.clone());
        }
        if let Some(prompt) = &self.prompt {
            if !prompt.is_empty() {
                form = form.text("prompt", prompt.clone());
            }
        }
        let request = self
            .client
            .post(format!("{}/audio/transcriptions", self.base_url))
            .multipart(form);
        let response = with_auth(request, &self.key)
            .send()
            .context("calling the transcription service")?;
        let response = check_status(response, "the transcription service")?;
        let payload: serde_json::Value = response.json().context("decoding the transcript")?;
        let text = payload
            .get("text")
            .and_then(serde_json::Value::as_str)
            .context("the transcription service returned no `text`")?
            .trim()
            .to_string();
        Ok(text)
    }
}

/// A local transcriber driven by argv: `{file}` is the audio file, `{model}` and `{language}`
/// the configured values. The transcript is read from stdout.
///
/// This is the provider that keeps audio on the machine, so it is the default in the example
/// config.
pub struct CommandStt {
    command: Vec<String>,
    language: String,
}

impl CommandStt {
    pub fn new(command: Vec<String>, language: String) -> Self {
        Self { command, language }
    }
}

impl SttProvider for CommandStt {
    fn name(&self) -> String {
        "command".to_string()
    }

    fn transcribe(&self, audio: &[u8], filename: &str) -> Result<String> {
        if self.command.is_empty() {
            bail!("stt.provider = command needs stt.command, and it is empty");
        }
        let dir = std::env::temp_dir().join(format!(
            "hey-jev-stt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).context("creating a temporary directory for audio")?;
        let file = dir.join(filename);
        std::fs::write(&file, audio).context("writing the audio file")?;

        let argv: Vec<String> = self
            .command
            .iter()
            .map(|part| {
                part.replace("{file}", &file.to_string_lossy())
                    .replace("{language}", &self.language)
            })
            .collect();
        let output = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .with_context(|| format!("running the transcription command `{}`", argv[0]))?;
        let _ = std::fs::remove_dir_all(&dir);

        if !output.status.success() {
            bail!(
                "the transcription command failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let text = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(text)
    }
}
