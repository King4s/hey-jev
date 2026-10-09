//! Finding a website in a spoken sentence.
//!
//! "open youtube dot com" is matched by pattern, because the domain is right there in the words.
//! "open the BBC" is not, so it goes to the language model — but only if the config gives it a
//! prompt, and only for a reply that looks like a URL. Anything else means the assistant asks
//! which site instead of opening a guess.

use anyhow::Result;
use regex::Regex;
use std::sync::OnceLock;

use crate::config::Config;
use crate::providers::LlmProvider;

/// A URL found in the text, or `None`.
pub fn find_url(text: &str, config: &Config, llm: Option<&dyn LlmProvider>) -> Result<Option<String>> {
    let spoken = spoken_dots(text, &config.web.dot_word);
    if let Some(url) = match_pattern(&spoken, &config.web.tlds) {
        return Ok(Some(url));
    }
    if config.web.prompt.is_empty() {
        return Ok(None);
    }
    let Some(llm) = llm else {
        return Ok(None);
    };
    let answer = llm.answer(&format!("{}\n\n{}", config.web.prompt, text))?;
    let answer = answer.trim();
    // Only a URL is acceptable. A model that answers with prose must not become an address.
    if !answer.starts_with("http") {
        return Ok(None);
    }
    let candidate = answer.split_whitespace().next().unwrap_or("").trim_end_matches(['.', ',', '!', '?']);
    Ok(Some(candidate.to_string()))
}

/// "youtube dot com" -> "youtube.com".
pub fn spoken_dots(text: &str, dot_word: &str) -> String {
    if dot_word.is_empty() {
        return text.to_string();
    }
    let pattern = format!(r"(?i)\s+\b{}\b\s+", regex::escape(dot_word));
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<String, Regex>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut cache = cache.lock().expect("regex cache");
    let regex = cache
        .entry(pattern.clone())
        .or_insert_with(|| Regex::new(&pattern).expect("escaped dot word compiles"));
    regex.replace_all(text, ".").to_string()
}

/// The first address in the text that ends in a known top-level domain.
fn match_pattern(text: &str, tlds: &[String]) -> Option<String> {
    if tlds.is_empty() {
        return None;
    }
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<Vec<String>, Regex>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut cache = cache.lock().expect("regex cache");
    let regex = cache.entry(tlds.to_vec()).or_insert_with(|| {
        let alternatives = tlds
            .iter()
            .map(|tld| regex::escape(tld))
            .collect::<Vec<_>>()
            .join("|");
        Regex::new(&format!(
            r"(?i)\b((?:[\w-]+\.)+(?:{alternatives})(?:/\S*)?)"
        ))
        .expect("escaped top-level domains compile")
    });
    let found = regex.captures(text)?.get(1)?.as_str();
    Some(format!(
        "https://{}",
        found.to_lowercase().trim_end_matches(['.', ',', '!', '?'])
    ))
}

/// How to name a site out loud: the host, without the scheme or a leading www.
pub fn site_name(url: &str) -> String {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let without_www = without_scheme.strip_prefix("www.").unwrap_or(without_scheme);
    without_www.split('/').next().unwrap_or(without_www).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EXAMPLE_CONFIG;

    fn config() -> Config {
        Config::from_toml(EXAMPLE_CONFIG).unwrap()
    }

    #[test]
    fn a_spoken_address_is_matched_without_asking_anyone() {
        let config = config();
        let url = find_url("open youtube dot com in chrome", &config, None).unwrap();
        assert_eq!(url.as_deref(), Some("https://youtube.com"));
        let url = find_url("open bbc.co.uk", &config, None).unwrap();
        assert_eq!(url.as_deref(), Some("https://bbc.co.uk"));
    }

    /// A sentence with no address and no model configured must not produce a guess.
    #[test]
    fn no_address_and_no_model_means_no_url() {
        let config = config();
        assert!(find_url("open the bbc", &config, None).unwrap().is_none());
    }

    /// Without a language-model prompt the fallback is off, not silently attempted.
    #[test]
    fn the_model_fallback_can_be_turned_off() {
        let mut config = config();
        config.web.prompt = String::new();
        assert!(find_url("open the bbc", &config, None).unwrap().is_none());
    }

    #[test]
    fn a_site_is_named_without_the_scheme() {
        assert_eq!(site_name("https://www.youtube.com/watch?v=1"), "youtube.com");
        assert_eq!(site_name("http://bbc.co.uk"), "bbc.co.uk");
    }

    /// The dot word is configurable, which is the whole point of it being config.
    #[test]
    fn the_dot_word_comes_from_config() {
        assert_eq!(spoken_dots("a dot b", "dot"), "a.b");
        assert_eq!(spoken_dots("a dot b", "punkt"), "a dot b");
    }
}
