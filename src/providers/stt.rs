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

/// Audio sent inside a chat message, which is how a router that accepts audio is called.
///
/// OpenRouter's `typesafe/jev-router` takes audio as an input and returns text, but it is a chat
/// model: it is not reachable through the `/audio/transcriptions` surface that
/// `openai_compatible` uses. This posts the audio as an `input_audio` content part and reads the
/// transcript out of the reply.
///
/// The distinction from `openai_compatible` is the protocol, not the vendor — any chat model that
/// accepts audio parts is reachable this way.
pub struct ChatCompletionsStt {
    client: reqwest::blocking::Client,
    base_url: String,
    key: Option<String>,
    model: String,
    language: String,
    prompt: Option<String>,
}

impl ChatCompletionsStt {
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

    /// What the model is told to do with the audio.
    fn instruction(&self) -> String {
        match &self.prompt {
            Some(prompt) if !prompt.is_empty() => prompt.clone(),
            _ => {
                let language = if self.language.is_empty() {
                    String::new()
                } else {
                    format!("The audio is in {}. ", self.language)
                };
                format!(
                    "{language}Transcribe this audio word for word. Reply with the transcript \
                     only, with no commentary and no quotation marks."
                )
            }
        }
    }
}

impl SttProvider for ChatCompletionsStt {
    fn name(&self) -> String {
        format!("chat_completions:{}", self.model)
    }

    fn transcribe(&self, audio: &[u8], _filename: &str) -> Result<String> {
        if self.model.trim().is_empty() {
            bail!("stt.model is required by this provider but is empty in the config");
        }
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD.encode(audio);
        let body = chat_audio_body(&self.model, &self.instruction(), &data);
        let request = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&body);
        let response = with_auth(request, &self.key)
            .send()
            .context("calling the transcription model")?;
        let response = check_status(response, "the transcription model")?;
        let payload: serde_json::Value =
            response.json().context("decoding the transcript")?;
        let text = payload["choices"][0]["message"]["content"]
            .as_str()
            .context("the transcription model returned no message content")?
            .trim()
            .to_string();
        Ok(text)
    }
}

/// The request body for audio riding inside a chat message.
///
/// Separate and pure so the shape can be checked without a network call: this is the part that is
/// easy to get subtly wrong, and a wrong content part is rejected or, worse, ignored.
pub fn chat_audio_body(model: &str, instruction: &str, audio_base64: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}},
            ],
        }],
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The audio must go over as an `input_audio` content part next to a text instruction, not as
    /// a bare message body — a router that accepts audio reads it from the content array.
    #[test]
    fn audio_is_sent_as_a_chat_content_part() {
        let body = chat_audio_body("typesafe/jev-router", "Transcribe this.", "QUJD");
        assert_eq!(body["model"], "typesafe/jev-router");
        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Transcribe this.");
        assert_eq!(content[1]["type"], "input_audio");
        assert_eq!(content[1]["input_audio"]["data"], "QUJD");
        assert_eq!(content[1]["input_audio"]["format"], "wav");
    }

    /// With no configured prompt the built-in instruction must carry the language, or a
    /// non-English recording is transcribed into the wrong one.
    #[test]
    fn the_default_instruction_names_the_language() {
        let provider = ChatCompletionsStt::new(
            reqwest::blocking::Client::new(),
            "https://example.invalid".into(),
            None,
            "m".into(),
            "da".into(),
            None,
        );
        let instruction = provider.instruction();
        assert!(instruction.contains("da"), "got {instruction:?}");
        assert!(instruction.to_lowercase().contains("transcribe"));

        // A configured prompt is used verbatim instead.
        let configured = ChatCompletionsStt::new(
            reqwest::blocking::Client::new(),
            "https://example.invalid".into(),
            None,
            "m".into(),
            "da".into(),
            Some("Write it down.".into()),
        );
        assert_eq!(configured.instruction(), "Write it down.");
    }
}
