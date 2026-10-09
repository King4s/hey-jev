//! A language model behind any OpenAI-compatible `/chat/completions` endpoint.
//!
//! This one implementation covers OpenRouter, OpenAI, a local LM Studio, Ollama and the fleet's
//! own AI gateway, because they all speak the same shape: which one is a matter of `base_url`,
//! `model` and `key_env` in the config.

use anyhow::{Context, Result};
use serde_json::json;

use super::LlmProvider;

pub struct OpenAiCompatibleLlm {
    client: reqwest::blocking::Client,
    base_url: String,
    key: Option<String>,
    model: String,
    max_tokens: u32,
    system_prompt: String,
}

impl OpenAiCompatibleLlm {
    pub fn new(
        client: reqwest::blocking::Client,
        base_url: String,
        key: Option<String>,
        model: String,
        max_tokens: u32,
        system_prompt: String,
    ) -> Self {
        Self { client, base_url: base_url.trim_end_matches('/').to_string(), key, model, max_tokens, system_prompt }
    }
}

impl LlmProvider for OpenAiCompatibleLlm {
    fn name(&self) -> String {
        format!("openai_compatible:{}", self.model)
    }

    fn answer(&self, text: &str) -> Result<String> {
        let body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
        });
        let request = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&body);
        let response = super::with_auth(request, &self.key)
            .send()
            .context("calling the language model")?;
        let response = super::check_status(response, "the language model")?;
        let json: serde_json::Value = response.json().context("decoding the language model reply")?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .context("the language model returned no message content")?
            .trim()
            .to_string();
        if content.is_empty() {
            anyhow::bail!("the language model returned an empty answer");
        }
        Ok(content)
    }
}
