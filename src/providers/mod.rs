//! The provider layer.
//!
//! Four jobs are done by something outside this program: deciding what a sentence means
//! (intent), answering a question (llm), speaking (tts), and turning speech into text (stt).
//! Which implementation does each job, at which endpoint, with which model, is configuration —
//! `provider = "..."` in the matching section. Adding an implementation means adding a variant
//! here and a name to the match; nothing else in the crate needs to know.

mod intent;
mod llm;
mod stt;
pub mod tts;

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{bail, Context, Result};

use crate::apps::AppList;
use crate::config::{Config, Question};
use crate::intent::Answers;
use crate::secrets::Secrets;

pub use intent::{Decisions, OpenAiCompatibleIntent};
pub use llm::OpenAiCompatibleLlm;
pub use stt::{CommandStt, OpenAiCompatibleStt};
pub use tts::{CommandTts, FishTts, SilentTts, SpeechTts};

/// Decides what a sentence means.
pub trait IntentProvider: Send + Sync {
    fn name(&self) -> String;
    /// Answer every question in one call. `apps` is the app key list, needed by questions whose
    /// options come from the app list rather than from the config.
    fn evaluate(
        &self,
        state: &str,
        questions: &BTreeMap<String, Question>,
        apps: &AppList,
    ) -> Result<Answers>;
}

/// Answers a factual question in one spoken sentence.
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> String;
    fn answer(&self, text: &str) -> Result<String>;
}

/// Turns a line of text into audio bytes.
pub trait TtsProvider: Send + Sync {
    fn name(&self) -> String;
    /// The audio container the bytes are in, used as the cache file extension.
    fn format(&self) -> &str;
    fn synthesize(&self, text: &str) -> Result<Vec<u8>>;
}

/// Turns recorded audio into text.
pub trait SttProvider: Send + Sync {
    fn name(&self) -> String;
    fn transcribe(&self, audio: &[u8], filename: &str) -> Result<String>;
}

/// An HTTP client with the configured timeout, shared by every HTTP provider.
pub fn http_client(timeout_secs: u64) -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .user_agent(concat!("hey-jev/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("building the HTTP client")
}

/// Build the intent provider named in `[intent] provider`.
///
/// `typesafe` and `decider` are the same implementation: the two vendors speak the same
/// decisions protocol, and only the endpoint, the model and the key differ. `perplexity` is
/// accepted as a spelling of `decider` because that is the company's name.
pub fn build_intent(config: &Config, secrets: &mut Secrets) -> Result<Box<dyn IntentProvider>> {
    let client = http_client(config.general.timeout_secs)?;
    let mut key = || secrets_maybe(secrets, &config.intent.key_env);
    match config.intent.provider.as_str() {
        "typesafe" | "decider" | "perplexity" => {
            let label = if config.intent.provider == "typesafe" { "typesafe" } else { "decider" };
            let key = secrets.get(&config.intent.key_env).with_context(|| {
                format!("intent.provider = {} needs a value for {}", config.intent.provider, config.intent.key_env)
            })?;
            Ok(Box::new(Decisions::new(
                client,
                config.intent.base_url.clone(),
                key,
                config.intent.model.clone(),
                label,
            )))
        }
        "openai_compatible" => Ok(Box::new(OpenAiCompatibleIntent::new(
            client,
            config.intent.base_url.clone(),
            key(),
            config.intent.model.clone(),
        ))),
        other => bail!(
            "unknown intent.provider `{other}` — known providers: typesafe, decider, openai_compatible"
        ),
    }
}

/// Build the language model named in `[llm] provider`, or `None` when the section is absent.
pub fn build_llm(config: &Config, secrets: &mut Secrets) -> Result<Option<Box<dyn LlmProvider>>> {
    let Some(llm) = &config.llm else {
        return Ok(None);
    };
    let client = http_client(config.general.timeout_secs)?;
    match llm.provider.as_str() {
        "openai_compatible" => Ok(Some(Box::new(OpenAiCompatibleLlm::new(
            client,
            llm.base_url.clone(),
            secrets_maybe(secrets, &llm.key_env),
            llm.model.clone(),
            llm.max_tokens,
            llm.system_prompt.clone(),
        )))),
        other => bail!("unknown llm.provider `{other}` — known providers: openai_compatible"),
    }
}

/// Build the speech provider named in `[tts] provider`.
pub fn build_tts(config: &Config, secrets: &mut Secrets) -> Result<Box<dyn TtsProvider>> {
    let client = http_client(config.general.timeout_secs)?;
    let tts = &config.tts;
    match tts.provider.as_str() {
        "fish" => {
            let key = secrets
                .get(&tts.key_env)
                .with_context(|| format!("tts.provider = fish needs a value for {}", tts.key_env))?;
            Ok(Box::new(FishTts::new(
                client,
                tts.base_url.clone(),
                key,
                tts.voice.clone(),
                tts.model.clone(),
                tts.format.clone(),
            )))
        }
        "openai_compatible" => Ok(Box::new(SpeechTts::new(
            client,
            tts.base_url.clone(),
            secrets_maybe(secrets, &tts.key_env),
            tts.voice.clone(),
            tts.model.clone(),
            tts.format.clone(),
        ))),
        "command" => Ok(Box::new(CommandTts::new(
            tts.command.clone(),
            tts.format.clone(),
        ))),
        "none" => Ok(Box::new(SilentTts::new(tts.format.clone()))),
        other => bail!(
            "unknown tts.provider `{other}` — known providers: fish, openai_compatible, command, none"
        ),
    }
}

/// Build the transcriber named in `[stt] provider`, or `None` when the section is absent.
pub fn build_stt(config: &Config, secrets: &mut Secrets) -> Result<Option<Box<dyn SttProvider>>> {
    let Some(stt) = &config.stt else {
        return Ok(None);
    };
    let client = http_client(config.general.timeout_secs)?;
    match stt.provider.as_str() {
        "openai_compatible" => Ok(Some(Box::new(OpenAiCompatibleStt::new(
            client,
            stt.base_url.clone(),
            secrets_maybe(secrets, &stt.key_env),
            stt.model.clone(),
            stt.language.clone(),
            stt.prompt.clone(),
        )))),
        "command" => Ok(Some(Box::new(CommandStt::new(
            stt.command.clone(),
            stt.language.clone(),
        )))),
        "none" => Ok(None),
        other => bail!(
            "unknown stt.provider `{other}` — known providers: openai_compatible, command, none"
        ),
    }
}

/// Look up a key, treating "no key configured" and "key not set" the same: both mean the
/// provider will send an unauthenticated request, which some local endpoints accept.
fn secrets_maybe(secrets: &mut Secrets, name: &str) -> Option<String> {
    if name.is_empty() {
        None
    } else {
        secrets.get(name)
    }
}

/// Add the `Authorization` header when a key is present.
pub(crate) fn with_auth(
    request: reqwest::blocking::RequestBuilder,
    key: &Option<String>,
) -> reqwest::blocking::RequestBuilder {
    match key {
        Some(key) if !key.is_empty() => request.bearer_auth(key),
        _ => request,
    }
}

/// Turn a non-success response into an error that names the endpoint and quotes the body.
///
/// Without the body a 401 and a 422 look identical, and the original's troubleshooting note
/// ("one of your keys is wrong") is then a guess rather than a reading.
pub(crate) fn check_status(response: reqwest::blocking::Response, what: &str) -> Result<reqwest::blocking::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().unwrap_or_default();
    let body = body.trim();
    let body = if body.len() > 400 { &body[..400] } else { body };
    bail!("{what} failed: HTTP {status} {body}")
}
