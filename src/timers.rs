//! Timers and reminders.
//!
//! A spoken duration is parsed against the config's own vocabulary, so "twenty five minutes",
//! "half an hour" and "a couple of minutes" all work and the words are not baked into the code.
//! The countdown is kept in memory: a timer does not survive the process, which is what the
//! original did too, and it means a restart never fires a stale alarm.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use regex::Regex;
use std::sync::OnceLock;

use crate::config::Timers as Vocab;

/// One running countdown.
#[derive(Debug, Clone)]
pub struct Running {
    /// When it fires.
    end: Instant,
    /// How long it was set for, which is what it is called when it has no label.
    pub secs: u64,
    /// What to remind about, when it is a reminder rather than a plain timer.
    pub label: Option<String>,
}

impl Running {
    pub fn remaining(&self) -> u64 {
        self.end.saturating_duration_since(Instant::now()).as_secs()
    }

    /// How it is named when speaking about it: its label, or "5 min timer".
    pub fn name(&self, vocab: &Vocab) -> String {
        match &self.label {
            Some(label) if !label.is_empty() => {
                let mut chars = label.chars();
                match chars.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    None => label.clone(),
                }
            }
            _ => format!("{} timer", short_duration(self.secs, vocab)),
        }
    }
}

/// The set of running timers, shared with the scheduler.
#[derive(Clone)]
pub struct Timers {
    inner: Arc<Mutex<Vec<Running>>>,
}

impl Default for Timers {
    fn default() -> Self {
        Self::new()
    }
}

impl Timers {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(Vec::new())) }
    }

    /// Start a countdown. The list stays sorted soonest-first, which is what "how long is left"
    /// and "cancel the timer" both mean.
    pub fn add(&self, secs: u64, label: Option<String>) {
        let mut timers = self.inner.lock().expect("timers lock");
        timers.push(Running {
            end: Instant::now() + Duration::from_secs(secs),
            secs,
            label,
        });
        timers.sort_by_key(|timer| timer.end);
    }

    pub fn count(&self) -> usize {
        self.inner.lock().expect("timers lock").len()
    }

    /// The soonest timer: its name and the seconds left.
    pub fn soonest(&self, vocab: &Vocab) -> Option<(String, u64)> {
        let timers = self.inner.lock().expect("timers lock");
        timers.first().map(|timer| (timer.name(vocab), timer.remaining()))
    }

    /// Cancel the most recently set timer, or every timer. Returns the number cancelled.
    pub fn cancel(&self, all: bool) -> usize {
        let mut timers = self.inner.lock().expect("timers lock");
        if timers.is_empty() {
            return 0;
        }
        if all {
            let count = timers.len();
            timers.clear();
            return count;
        }
        // The one set most recently is the one that has run the longest, because the list is
        // sorted by when it fires.
        match timers.iter().enumerate().max_by_key(|(_, timer)| timer.secs - timer.remaining()) {
            Some((index, _)) => {
                timers.remove(index);
                1
            }
            None => 0,
        }
    }

    /// Remove and return every timer that is due.
    pub fn take_due(&self) -> Vec<Running> {
        let now = Instant::now();
        let mut timers = self.inner.lock().expect("timers lock");
        let due: Vec<Running> = timers.iter().filter(|timer| timer.end <= now).cloned().collect();
        timers.retain(|timer| timer.end > now);
        due
    }
}

// --------------------------------------------------------------------------- parsing

/// The duration pattern, built from the configured units so the words come from the config.
fn duration_regex(vocab: &Vocab) -> Regex {
    static CACHE: OnceLock<Mutex<BTreeMap<Vec<String>, Regex>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut units: Vec<String> = vocab.units.keys().cloned().collect();
    // Longest first, so "minutes" is tried before "m".
    units.sort_by_key(|unit| std::cmp::Reverse(unit.len()));
    let key = units.clone();

    let mut cache = cache.lock().expect("regex cache");
    cache
        .entry(key)
        .or_insert_with(|| {
            let alternatives = units
                .iter()
                .map(|unit| regex::escape(unit))
                .collect::<Vec<_>>()
                .join("|");
            Regex::new(&format!(
                r"(?i)(\d+(?:\.\d+)?)\s*({alternatives})\b(\s+and\s+a\s+half)?"
            ))
            .expect("escaped units compile")
        })
        .clone()
}

/// Replace spoken numbers with digits, and "half an hour" with its seconds.
///
/// This is the part that has to know the vocabulary: "twenty five minutes" is two words for one
/// number, and only the config says which words those are.
pub fn normalize_numbers(text: &str, vocab: &Vocab) -> String {
    let mut out = text.to_lowercase();

    // "half an hour" / "half a minute" -> half that unit, expressed in seconds. The original
    // hardcoded "half an hour = 30 minutes"; deriving it keeps the arithmetic right for every
    // unit the config defines, including ones that are not a whole number of minutes.
    let second_word = vocab
        .units
        .iter()
        .find(|(_, seconds)| **seconds == 1)
        .map(|(word, _)| word.clone())
        .unwrap_or_else(|| "seconds".to_string());
    for (unit, seconds) in &vocab.units {
        let pattern = format!(r"(?i)\bhalf\s+an?\s+{}\b", regex::escape(unit));
        if let Ok(regex) = Regex::new(&pattern) {
            if regex.is_match(&out) {
                out = regex
                    .replace_all(&out, format!("{} {}", seconds / 2, second_word))
                    .to_string();
            }
        }
    }
    out = out.replace("a couple of", "couple");

    // "a minute" / "few minutes" / "couple of hours" -> the number word alone, then digits.
    let words: Vec<&str> = vocab.numbers.keys().map(String::as_str).collect();
    if !words.is_empty() {
        let alternatives = words
            .iter()
            .map(|word| regex::escape(word))
            .collect::<Vec<_>>()
            .join("|");
        let pattern = format!(r"(?i)\b({alternatives})\s+({})\b",
            vocab.units.keys().map(|u| regex::escape(u)).collect::<Vec<_>>().join("|"));
        if let Ok(regex) = Regex::new(&pattern) {
            out = regex
                .replace_all(&out, |caps: &regex::Captures| {
                    let word = &caps[1];
                    let unit = &caps[2];
                    if matches!(word, "a" | "an" | "few" | "couple") {
                        if let Some(value) = vocab.numbers.get(word) {
                            return format!("{value} {unit}");
                        }
                    }
                    format!("{word} {unit}")
                })
                .to_string();
        }
    }

    // Spelled-out numbers, including the "twenty five" pairing.
    let mut result: Vec<String> = Vec::new();
    let tokens: Vec<&str> = out.split_whitespace().collect();
    let mut index = 0;
    while index < tokens.len() {
        let token = tokens[index].trim_matches(|c: char| ".,!?".contains(c));
        let next = tokens
            .get(index + 1)
            .map(|t| t.trim_matches(|c: char| ".,!?".contains(c)));
        // "and a half" is a duration the regex understands; the article must not become a 1.
        if matches!(token, "a" | "an") && next == Some("half") {
            result.push(tokens[index].to_string());
            index += 1;
            continue;
        }
        match vocab.numbers.get(token) {
            Some(value) => {
                let mut value = *value;
                if value >= 20 {
                    if let Some(second) = next.and_then(|word| vocab.numbers.get(word)) {
                        if *second < 10 && *second > 0 {
                            value += second;
                            index += 1;
                        }
                    }
                }
                result.push(value.to_string());
            }
            None => result.push(tokens[index].to_string()),
        }
        index += 1;
    }
    result.join(" ")
}

/// The seconds in a unit, as a multiplier for a half.
#[allow(dead_code)]
fn unit_divisor(seconds: u32) -> u32 {
    if seconds >= 3600 {
        3600
    } else if seconds >= 60 {
        60
    } else {
        1
    }
}

/// The total seconds mentioned in a sentence, or `None`.
pub fn parse_duration(text: &str, vocab: &Vocab) -> Option<u64> {
    let normalized = normalize_numbers(text, vocab);
    let regex = duration_regex(vocab);
    let mut total = 0.0;
    for caps in regex.captures_iter(&normalized) {
        let amount: f64 = caps[1].parse().ok()?;
        let unit = caps[2].to_lowercase();
        let seconds = *vocab.units.get(&unit)? as f64;
        total += amount * seconds;
        if caps.get(3).is_some() {
            total += seconds / 2.0;
        }
    }
    let total = total.round() as u64;
    (total > 0).then_some(total)
}

/// What to remind about: the part after "to", minus any duration.
///
/// "remind me in 5 minutes to call mum" -> "call mum". `to_word` is the configuration's word for
/// "to", so the sentence structure is a setting too.
pub fn parse_reminder(text: &str, vocab: &Vocab, to_word: &str) -> Option<String> {
    let normalized = normalize_numbers(text, vocab);
    let pattern = format!(r"(?i)\b{}\s+(.+)$", regex::escape(to_word));
    let regex = Regex::new(&pattern).ok()?;
    let tail = regex.captures(&normalized)?.get(1)?.as_str();
    let without_duration = duration_regex(vocab).replace_all(tail, "");
    let cleaned = without_duration
        .split_whitespace()
        .filter(|word| !matches!(*word, "please" | "in" | "for" | "after"))
        .collect::<Vec<_>>()
        .join(" ");
    let cleaned = cleaned.trim_matches(|c: char| " .,!?".contains(c)).to_string();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// A duration as spoken words, using the config's own unit names.
pub fn say_duration(secs: u64, vocab: &Vocab) -> String {
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    let mut parts: Vec<String> = Vec::new();
    for (value, unit) in [(hours, 3600), (minutes, 60), (seconds, 1)] {
        if value == 0 {
            continue;
        }
        parts.push(match vocab.spoken.iter().find(|spoken| spoken.seconds == unit) {
            Some(spoken) => {
                let word = if value == 1 { &spoken.one } else { &spoken.many };
                format!("{value} {word}")
            }
            // A unit with no spoken form still counts correctly, it just reads as a number.
            None => format!("{value} seconds"),
        });
    }
    // Once it is hours long, the seconds are noise.
    if hours > 0 && parts.len() > 2 {
        parts.truncate(2);
    } else if minutes >= 10 && parts.len() > 1 && hours == 0 {
        parts.truncate(1);
    }
    if parts.is_empty() {
        return "no time".to_string();
    }
    parts.join(" and ")
}

/// A duration as a short label: "5 min", "1 hr 30 min".
pub fn short_duration(secs: u64, vocab: &Vocab) -> String {
    let mut units: Vec<u32> = vocab.spoken.iter().map(|spoken| spoken.seconds).collect();
    units.sort_unstable_by(|a, b| b.cmp(a));
    let mut remaining = secs;
    let mut parts = Vec::new();
    for unit in units {
        if unit == 0 {
            continue;
        }
        let count = remaining / unit as u64;
        if count > 0 {
            let word = vocab
                .spoken
                .iter()
                .find(|spoken| spoken.seconds == unit)
                .map(|spoken| {
                    if count == 1 { spoken.one.clone() } else { spoken.many.clone() }
                })
                .unwrap_or_else(|| format!("{unit}s"));
            // Short labels abbreviate; the long form is only for speech.
            let short = word.chars().take(3).collect::<String>();
            parts.push(format!("{count} {short}"));
            remaining -= count * unit as u64;
        }
    }
    if parts.is_empty() {
        return "0 sec".to_string();
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EXAMPLE_CONFIG;

    fn vocab() -> Vocab {
        crate::config::Config::from_toml(EXAMPLE_CONFIG).unwrap().timers.unwrap()
    }

    #[test]
    fn a_spoken_duration_is_parsed_in_every_shape() {
        let vocab = vocab();
        assert_eq!(parse_duration("set a timer for 5 minutes", &vocab), Some(300));
        assert_eq!(parse_duration("set a timer for twenty five minutes", &vocab), Some(1500));
        assert_eq!(parse_duration("remind me in 20 minutes", &vocab), Some(1200));
        assert_eq!(parse_duration("half an hour", &vocab), Some(1800));
        assert_eq!(parse_duration("a couple of minutes", &vocab), Some(120));
        assert_eq!(parse_duration("2 hours and a half", &vocab), Some(9000));
        assert_eq!(parse_duration("90 seconds", &vocab), Some(90));
        assert_eq!(parse_duration("no duration here", &vocab), None);
        assert_eq!(parse_duration("set a timer for 0 minutes", &vocab), None);
    }

    #[test]
    fn a_reminder_keeps_what_to_remind_about_and_drops_the_duration() {
        let vocab = vocab();
        assert_eq!(
            parse_reminder("remind me in 5 minutes to call mum", &vocab, "to").as_deref(),
            Some("call mum")
        );
        assert_eq!(
            parse_reminder("remind me to take the bins out", &vocab, "to").as_deref(),
            Some("take the bins out")
        );
        assert_eq!(parse_reminder("set a timer for 5 minutes", &vocab, "to"), None);
    }

    /// What it says must use the configured unit names, not hardcoded English.
    #[test]
    fn a_duration_is_spoken_with_the_configured_words() {
        let vocab = vocab();
        assert_eq!(say_duration(300, &vocab), "5 minutes");
        assert_eq!(say_duration(60, &vocab), "1 minute");
        assert_eq!(say_duration(69, &vocab), "1 minute and 9 seconds");
        assert_eq!(say_duration(3600, &vocab), "1 hour");
        assert_eq!(say_duration(0, &vocab), "no time");
        // Over ten minutes, the seconds are dropped from speech.
        assert_eq!(say_duration(605, &vocab), "10 minutes");
    }

    #[test]
    fn timers_run_and_are_cancelled_soonest_first() {
        let timers = Timers::new();
        assert!(timers.soonest(&vocab()).is_none());
        timers.add(300, None);
        timers.add(60, Some("call mum".into()));
        assert_eq!(timers.count(), 2);
        let (name, left) = timers.soonest(&vocab()).unwrap();
        assert_eq!(name, "Call mum");
        assert!(left <= 60);
        // Not "all" cancels the most recently set one, which is the 300-second timer.
        assert_eq!(timers.cancel(false), 1);
        assert_eq!(timers.count(), 1);
        assert_eq!(timers.cancel(true), 1);
        assert_eq!(timers.cancel(true), 0);
    }

    /// A timer with no label is named by its length, and a short duration is abbreviated.
    #[test]
    fn an_unlabelled_timer_is_named_by_its_length() {
        let timers = Timers::new();
        timers.add(1500, None);
        assert_eq!(timers.soonest(&vocab()).unwrap().0, "25 min timer");
    }
}
