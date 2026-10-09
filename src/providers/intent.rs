//! Intent providers: what does this sentence mean?
//!
//! Two implementations. `typesafe` calls TypeSafe's System One evaluation endpoint, which
//! returns calibrated probabilities directly. `openai_compatible` asks any chat model for the
//! same shape and then *verifies* it, because a chat model will happily return a probability
//! distribution over options that do not exist, or one that does not sum to 1.
//!
//! The verification is the point. A fabricated distribution that passes through unchecked is
//! indistinguishable, downstream, from a calibrated one — and it is the downstream code that
//! decides whether to launch an application.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::apps::AppList;
use crate::config::{Question, QuestionKind};
use crate::intent::{Answer, Answers};

use super::{check_status, with_auth, IntentProvider};

/// How far a probability distribution may drift from 1 before it is rejected.
const PROBABILITY_TOLERANCE: f64 = 0.02;
/// How far the chosen option may sit from the highest-probability option.
const ARGMAX_TOLERANCE: f64 = 1e-6;

/// The options a question offers, in a stable order: configured first, then the app list, then
/// the "nothing here" option.
pub fn options_for(question: &Question, apps: &AppList, none_option: &str) -> Vec<String> {
    let mut options: Vec<String> = question.criteria.keys().cloned().collect();
    if question.include_apps {
        for key in apps.keys() {
            if !options.contains(&key) {
                options.push(key);
            }
        }
    }
    if question.include_none && !options.iter().any(|option| option == none_option) {
        options.push(none_option.to_string());
    }
    options
}

/// The question map as the wire protocol wants it: a choice sends a map of option to
/// description, a score sends an ordered array of levels.
pub fn questions_payload(
    questions: &BTreeMap<String, Question>,
    apps: &AppList,
    none_option: &str,
) -> Value {
    let mut out = Map::new();
    for (id, question) in questions {
        let instructions = question
            .instructions
            .replace("{apps}", &apps.names.len().to_string());
        let mut entry = Map::new();
        match question.kind {
            QuestionKind::Choice => {
                entry.insert("type".into(), json!("choice"));
                entry.insert("instructions".into(), json!(instructions));
                let mut criteria = Map::new();
                for option in options_for(question, apps, none_option) {
                    // An option with no description is sent as a null, which is how the
                    // original marked "this option needs no explanation".
                    let description = question.criteria.get(&option).map(String::as_str);
                    let value = match description {
                        Some(text) if !text.is_empty() => json!(text),
                        _ => Value::Null,
                    };
                    criteria.insert(option, value);
                }
                entry.insert("criteria".into(), Value::Object(criteria));
            }
            QuestionKind::Noul => {
                entry.insert("type".into(), json!("noul"));
                entry.insert("instructions".into(), json!(instructions));
                if !question.criteria.is_empty() {
                    let mut criteria = Map::new();
                    for (key, description) in &question.criteria {
                        criteria.insert(key.clone(), json!(description));
                    }
                    entry.insert("criteria".into(), Value::Object(criteria));
                }
            }
            QuestionKind::Score => {
                entry.insert("type".into(), json!("score"));
                entry.insert("instructions".into(), json!(instructions));
                entry.insert("criteria".into(), json!(question.levels));
            }
        }
        out.insert(id.clone(), Value::Object(entry));
    }
    Value::Object(out)
}

// --------------------------------------------------------------------------- TypeSafe

pub struct TypeSafeIntent {
    client: reqwest::blocking::Client,
    url: String,
    key: String,
    model: String,
}

impl TypeSafeIntent {
    pub fn new(
        client: reqwest::blocking::Client,
        url: String,
        key: String,
        model: String,
    ) -> Self {
        Self { client, url, key, model }
    }
}

impl IntentProvider for TypeSafeIntent {
    fn name(&self) -> String {
        format!("typesafe:{}", self.model)
    }

    fn evaluate(
        &self,
        state: &str,
        questions: &BTreeMap<String, Question>,
        apps: &AppList,
    ) -> Result<Answers> {
        // TypeSafe needs the "nothing here" option name too; it is read back off the questions
        // that ask for it, so the provider does not need its own copy of the config.
        let none_option = "none";
        let body = json!({
            "model": self.model,
            "state": state,
            "questions": questions_payload(questions, apps, none_option),
        });
        let request = self.client.post(&self.url).json(&body);
        let response = with_auth(request, &Some(self.key.clone()))
            .send()
            .context("calling TypeSafe")?;
        let response = check_status(response, "TypeSafe")?;
        let payload: Value = response.json().context("decoding TypeSafe's reply")?;
        let answers = payload
            .get("answers")
            .and_then(Value::as_object)
            .context("TypeSafe's reply had no `answers` object")?;
        parse_typesafe_answers(answers)
    }
}

fn parse_typesafe_answers(answers: &Map<String, Value>) -> Result<Answers> {
    let mut out = BTreeMap::new();
    for (id, answer) in answers {
        let kind = answer.get("type").and_then(Value::as_str).unwrap_or("choice");
        let parsed = match kind {
            "noul" => Answer {
                kind: kind.to_string(),
                noul: answer.get("noul").and_then(Value::as_f64),
                // A noul's confidence is its distance from a coin flip, per the API.
                confidence: answer
                    .get("noul")
                    .and_then(Value::as_f64)
                    .map(|p| p.max(1.0 - p).clamp(0.0, 1.0))
                    .unwrap_or(0.0),
                ..Default::default()
            },
            "score" => Answer {
                kind: kind.to_string(),
                score: answer.get("score").and_then(Value::as_f64),
                legend: parse_legend(answer.get("legend")),
                confidence: answer.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
                ..Default::default()
            },
            _ => Answer {
                kind: "choice".to_string(),
                choice: answer.get("choice").and_then(Value::as_str).map(str::to_string),
                confidence: answer.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
                ..Default::default()
            },
        };
        out.insert(id.clone(), parsed);
    }
    Ok(Answers { map: out })
}

fn parse_legend(value: Option<&Value>) -> BTreeMap<String, String> {
    value
        .and_then(Value::as_object)
        .map(|map| {
            map.iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|text| (key.clone(), text.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

// --------------------------------------------------------------------------- OpenAI-compatible

pub struct OpenAiCompatibleIntent {
    client: reqwest::blocking::Client,
    base_url: String,
    key: Option<String>,
    model: String,
}

impl OpenAiCompatibleIntent {
    pub fn new(
        client: reqwest::blocking::Client,
        base_url: String,
        key: Option<String>,
        model: String,
    ) -> Self {
        Self { client, base_url: base_url.trim_end_matches('/').to_string(), key, model }
    }
}

const SYSTEM_PROMPT: &str = "You decide what a spoken request means. \
You are given `questions`, each with an id, a type and its allowed answers, and `state`, the \
user's words. Answer every question. Reply with JSON only, no prose and no code fences, in \
this exact shape:\n\
{\"answers\": {\"<question id>\": {\"probabilities\": {\"<allowed answer>\": <probability>}}, ...}}\n\
For a `noul` question reply {\"noul\": <probability of yes>} instead of `probabilities`. \
Probabilities are numbers between 0 and 1 across exactly the allowed answers, and they must sum \
to 1. Use the full range: 0.99 when the words clearly say it, 0.5 when they do not.";

impl IntentProvider for OpenAiCompatibleIntent {
    fn name(&self) -> String {
        format!("openai_compatible:{}", self.model)
    }

    fn evaluate(
        &self,
        state: &str,
        questions: &BTreeMap<String, Question>,
        apps: &AppList,
    ) -> Result<Answers> {
        let none_option = "none";
        let schema = questions_payload(questions, apps, none_option);
        let body = json!({
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format!(
                    "questions:\n{}\n\nstate: {}",
                    serde_json::to_string_pretty(&schema)?,
                    state
                )},
            ],
        });
        let request = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&body);
        let response = with_auth(request, &self.key)
            .send()
            .context("calling the intent model")?;
        let response = check_status(response, "the intent model")?;
        let payload: Value = response.json().context("decoding the intent model's reply")?;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .context("the intent model returned no message content")?;
        let json = extract_json(content)
            .with_context(|| format!("the intent model did not return JSON: {content:.300}"))?;
        let answers = json
            .get("answers")
            .and_then(Value::as_object)
            .context("the intent model's JSON had no `answers` object")?;
        verify_answers(answers, questions, apps, none_option)
    }
}

/// Pull the outermost JSON object out of a reply, tolerating a model that wraps it in prose or
/// fences despite being told not to.
pub fn extract_json(text: &str) -> Option<Value> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&text[start..=end]).ok()
}

/// Check every answer against the question it claims to answer.
///
/// A malformed distribution is an error, not a retry: retrying the same question invites a
/// second fabrication, and the caller's job here is to fail loudly and ask the user again.
pub fn verify_answers(
    answers: &Map<String, Value>,
    questions: &BTreeMap<String, Question>,
    apps: &AppList,
    none_option: &str,
) -> Result<Answers> {
    let mut out = BTreeMap::new();
    for (id, raw) in answers {
        let question = questions.get(id).with_context(|| {
            format!("the intent model answered `{id}`, which was not asked")
        })?;
        let answer = match question.kind {
            QuestionKind::Noul => {
                let probability = raw
                    .get("noul")
                    .and_then(Value::as_f64)
                    .with_context(|| format!("`{id}` is a noul but came back without one"))?;
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    bail!("`{id}` returned a noul of {probability}, which is not a probability");
                }
                Answer {
                    kind: "noul".into(),
                    noul: Some(probability),
                    confidence: probability.max(1.0 - probability),
                    ..Default::default()
                }
            }
            QuestionKind::Score => {
                let levels = &question.levels;
                let probabilities = read_probabilities(raw, id)?;
                let (score, legend, confidence) = score_from(levels, &probabilities, id)?;
                Answer {
                    kind: "score".into(),
                    score: Some(score),
                    legend,
                    confidence,
                    ..Default::default()
                }
            }
            QuestionKind::Choice => {
                let allowed = options_for(question, apps, none_option);
                let probabilities = read_probabilities(raw, id)?;
                let (choice, confidence) = argmax(&allowed, &probabilities, id)?;
                Answer {
                    kind: "choice".into(),
                    choice: Some(choice),
                    confidence,
                    ..Default::default()
                }
            }
        };
        out.insert(id.clone(), answer);
    }
    Ok(Answers { map: out })
}

fn read_probabilities(raw: &Value, id: &str) -> Result<BTreeMap<String, f64>> {
    let map = raw
        .get("probabilities")
        .and_then(Value::as_object)
        .with_context(|| format!("`{id}` came back without a `probabilities` object"))?;
    let mut out = BTreeMap::new();
    for (option, value) in map {
        let probability = value
            .as_f64()
            .with_context(|| format!("`{id}`: probability for `{option}` is not a number"))?;
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            bail!("`{id}`: probability for `{option}` is {probability}, which is not between 0 and 1");
        }
        out.insert(option.clone(), probability);
    }
    if out.is_empty() {
        bail!("`{id}`: probabilities is empty");
    }
    let total: f64 = out.values().sum();
    if (total - 1.0).abs() > PROBABILITY_TOLERANCE {
        bail!("`{id}`: probabilities sum to {total}, not 1");
    }
    Ok(out)
}

/// The highest-probability option, after checking the distribution covers exactly the allowed
/// options. An option the model invented would otherwise become an action.
fn argmax(
    allowed: &[String],
    probabilities: &BTreeMap<String, f64>,
    id: &str,
) -> Result<(String, f64)> {
    let invented: Vec<&String> = probabilities
        .keys()
        .filter(|option| !allowed.contains(option))
        .collect();
    if !invented.is_empty() {
        bail!("`{id}`: probabilities mention {invented:?}, which are not options of this question");
    }
    let missing: Vec<&String> = allowed
        .iter()
        .filter(|option| !probabilities.contains_key(*option))
        .collect();
    if !missing.is_empty() {
        bail!("`{id}`: probabilities are missing {missing:?}");
    }
    let mut best: Option<(&String, f64)> = None;
    for option in allowed {
        let probability = probabilities[option];
        if best.map(|(_, top)| probability > top + ARGMAX_TOLERANCE).unwrap_or(true) {
            best = Some((option, probability));
        }
    }
    let (choice, confidence) = best.expect("allowed is non-empty: checked at load time");
    Ok((choice.clone(), confidence))
}

/// The probability-weighted level, its legend, and the distribution's concentration.
fn score_from(
    levels: &[String],
    probabilities: &BTreeMap<String, f64>,
    id: &str,
) -> Result<(f64, BTreeMap<String, String>, f64)> {
    for option in probabilities.keys() {
        if !levels.contains(option) {
            bail!("`{id}`: probabilities mention `{option}`, which is not a level of this question");
        }
    }
    let mut legend = BTreeMap::new();
    let mut score = 0.0;
    let mut top = 0.0_f64;
    for (index, level) in levels.iter().enumerate() {
        let probability = probabilities.get(level).copied().unwrap_or(0.0);
        legend.insert(index.to_string(), level.clone());
        score += index as f64 * probability;
        top = top.max(probability);
    }
    Ok((score, legend, top))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question(kind: QuestionKind) -> Question {
        Question {
            kind,
            instructions: "what".into(),
            criteria: BTreeMap::new(),
            levels: Vec::new(),
            include_apps: false,
            include_none: false,
        }
    }

    fn apps() -> AppList {
        let mut list = AppList::default();
        list.names.insert("spotify".into(), "Spotify".into());
        list.names.insert("slack".into(), "Slack".into());
        list
    }

    /// The "which app?" question must follow the app list instead of duplicating it.
    #[test]
    fn app_options_come_from_the_app_list() {
        let mut q = question(QuestionKind::Choice);
        q.include_apps = true;
        q.include_none = true;
        q.criteria.insert("none".into(), String::new());
        assert_eq!(options_for(&q, &apps(), "none"), vec!["none", "slack", "spotify"]);
    }

    /// An option with no description goes over the wire as a null, not as an empty string.
    #[test]
    fn empty_descriptions_become_nulls() {
        let mut q = question(QuestionKind::Choice);
        q.criteria.insert("open".into(), "open the app".into());
        q.criteria.insert("quit".into(), String::new());
        let mut questions = BTreeMap::new();
        questions.insert("app_action".into(), q);
        let payload = questions_payload(&questions, &AppList::default(), "none");
        assert_eq!(payload["app_action"]["criteria"]["open"], json!("open the app"));
        assert!(payload["app_action"]["criteria"]["quit"].is_null());
    }

    /// A score question's levels go over as an ordered array, which is what the API takes.
    #[test]
    fn score_levels_are_sent_as_an_array() {
        let mut q = question(QuestionKind::Score);
        q.levels = vec!["silent".into(), "quiet".into(), "loud".into()];
        let mut questions = BTreeMap::new();
        questions.insert("volume_level".into(), q);
        let payload = questions_payload(&questions, &AppList::default(), "none");
        assert_eq!(payload["volume_level"]["criteria"], json!(["silent", "quiet", "loud"]));
    }

    #[test]
    fn a_well_formed_distribution_is_accepted() {
        let mut q = question(QuestionKind::Choice);
        q.criteria.insert("open".into(), String::new());
        q.criteria.insert("quit".into(), String::new());
        let questions = BTreeMap::from([("app_action".to_string(), q)]);
        let answers = serde_json::json!({
            "app_action": {"probabilities": {"open": 0.9, "quit": 0.1}}
        });
        let verified =
            verify_answers(answers.as_object().unwrap(), &questions, &AppList::default(), "none")
                .unwrap();
        assert_eq!(verified.choice("app_action").0.as_deref(), Some("open"));
        assert!((verified.choice("app_action").1 - 0.9).abs() < 1e-9);
    }

    /// The three ways a chat model gets this wrong, and the option it would otherwise launch.
    #[test]
    fn a_fabricated_distribution_is_rejected() {
        let mut q = question(QuestionKind::Choice);
        q.criteria.insert("open".into(), String::new());
        q.criteria.insert("quit".into(), String::new());
        let questions = BTreeMap::from([("app_action".to_string(), q)]);
        let of = |value: Value| {
            verify_answers(value.as_object().unwrap(), &questions, &AppList::default(), "none")
                .map(|answers| answers.choice("app_action").0)
        };
        // An option that does not exist.
        assert!(of(serde_json::json!({
            "app_action": {"probabilities": {"open": 0.9, "delete": 0.1}}
        }))
        .is_err());
        // A distribution that does not sum to 1.
        assert!(of(serde_json::json!({
            "app_action": {"probabilities": {"open": 0.9, "quit": 0.9}}
        }))
        .is_err());
        // A missing option.
        assert!(of(serde_json::json!({
            "app_action": {"probabilities": {"open": 1.0}}
        }))
        .is_err());
        // A question that was never asked.
        assert!(of(serde_json::json!({
            "something_else": {"probabilities": {"open": 1.0}}
        }))
        .is_err());
        // A probability outside 0..1.
        assert!(of(serde_json::json!({
            "app_action": {"probabilities": {"open": 1.4, "quit": -0.4}}
        }))
        .is_err());
    }

    /// A score's level is read back out of the legend, and the score is the weighted index.
    #[test]
    fn a_score_distribution_becomes_a_level() {
        let mut q = question(QuestionKind::Score);
        q.levels = vec!["silent".into(), "quiet".into(), "loud".into()];
        let questions = BTreeMap::from([("volume_level".to_string(), q)]);
        let answers = serde_json::json!({
            "volume_level": {"probabilities": {"silent": 0.0, "quiet": 0.8, "loud": 0.2}}
        });
        let verified =
            verify_answers(answers.as_object().unwrap(), &questions, &AppList::default(), "none")
                .unwrap();
        let (level, _) = verified.score("volume_level");
        assert_eq!(level.as_deref(), Some("quiet"));
    }

    #[test]
    fn json_is_extracted_from_a_chatty_reply() {
        let json = extract_json("Sure! ```json\n{\"answers\": {}}\n```").unwrap();
        assert!(json.get("answers").is_some());
        assert!(extract_json("no json here").is_none());
    }
}
