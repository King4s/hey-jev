//! Configuration.
//!
//! Everything that was a module-level constant in the original Python lives here instead:
//! providers, models, voice, wake words, thresholds, prompts, paths and replies. Nothing in
//! this crate carries a setting that you cannot see and change in a config file.
//!
//! Defaults exist only for values where omitting the key is unambiguous and harmless (a
//! request timeout, a sample rate). Anything that decides *what the assistant does* —
//! which provider, which model, which questions, which replies — has no Rust default and is
//! required in the file, so `config.example.toml` stays the single definition of them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::apps::AppList;

/// The shipped example, written out by `--init`.
pub const EXAMPLE_CONFIG: &str = include_str!("../config.example.toml");

/// The filename looked for in the default locations.
pub const DEFAULT_FILENAME: &str = "config.toml";

/// A fully materialised configuration.
///
/// `deny_unknown_fields` is deliberate: a mistyped key is an error, not a silently ignored
/// setting. The usual failure mode of a config-driven program is a knob that looks set and is not.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub general: General,
    pub intent: Intent,
    pub tts: Tts,
    #[serde(default)]
    pub llm: Option<Llm>,
    #[serde(default)]
    pub stt: Option<Stt>,
    #[serde(default)]
    pub keys: Keys,
    #[serde(default)]
    pub apps: Apps,
    #[serde(default)]
    pub wake: Wake,
    #[serde(default)]
    pub replies: Replies,
    /// How a spoken sentence is turned into a website to open.
    pub web: Web,
    /// The number vocabulary for timers, and the only thing that makes the timer actions work.
    /// Absent means the assistant cannot set timers.
    #[serde(default)]
    pub timers: Option<Timers>,
    /// Where the config file itself was read from. Not part of the file.
    #[serde(skip)]
    pub source_path: Option<PathBuf>,
    /// The app list, read from `[apps] file` when the config is loaded. Not part of the file.
    #[serde(skip)]
    pub apps_list: AppList,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct General {
    /// `ptt` = hold a key to talk, `wake` = always listening for the wake word.
    pub mode: Mode,
    /// Which platform driver to use: `macos`, `linux`, `recording`, or `auto` for this host.
    #[serde(default)]
    pub platform: Option<String>,
    /// Confidence below which an answer is not acted on.
    pub gate: f64,
    /// Directory holding the log, the TTS cache and any saved clips.
    pub state_dir: String,
    /// Request timeout for every provider call, in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Actions whose reply must be spoken BEFORE the action runs, because the action takes the
    /// sound or the screen away and the assistant would otherwise talk to itself.
    #[serde(default)]
    pub speak_before: Vec<String>,
    /// How the named volume levels map to a number. Required, because the level names come from
    /// the intent config's score question and the numbers cannot be guessed from them.
    pub volume: Volume,
}

/// The numeric value of each named volume level.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Volume {
    /// How far one "turn it up" moves the volume.
    pub step: u8,
    /// Level name -> the volume to set.
    pub levels: BTreeMap<String, u8>,
}

impl Volume {
    /// The volume for a named level, or a message naming the missing key.
    pub fn level(&self, name: &str) -> Result<u8> {
        self.levels.get(name).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "volume level `{name}` has no value: add it to [general.volume] levels"
            )
        })
    }

    /// The largest volume any level can reach; used to clamp "turn it up".
    pub fn ceiling(&self) -> u8 {
        self.levels.values().copied().max().unwrap_or(100)
    }
}

fn default_timeout() -> u64 {
    30
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Ptt,
    Wake,
}

// --------------------------------------------------------------------------- intent

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Intent {
    /// Which implementation answers the questions: `typesafe` or `openai_compatible`.
    pub provider: String,
    /// The model name handed to that provider.
    pub model: String,
    /// Endpoint. For `typesafe` this is the System One evaluation URL.
    pub base_url: String,
    /// Name of the key to authenticate with, resolved by [`crate::secrets`].
    pub key_env: String,
    /// The questions asked in one fan-out call, keyed by the id answers come back under.
    pub questions: BTreeMap<String, Question>,
    /// Question ids that must be scoped "first"/"second" for a compound request. The same
    /// questions are asked twice under `<slot>_<id>`, which is how two actions in one
    /// sentence are told apart without a second model.
    #[serde(default)]
    pub split_slots: Vec<String>,
    /// Id of the yes/no question that decides whether the request holds two actions.
    pub compound_question: String,
    /// Question id naming the request kind (command / question / chit-chat / gibberish).
    pub category_question: String,
    /// Question id naming the primary thing being controlled.
    pub target_question: String,
    /// The option of `category_question` meaning "the user asked something", answered by an LLM.
    pub category_information: String,
    /// The option meaning "just talking", answered with a scripted reply.
    pub category_chit_chat: String,
    /// The option meaning "garbled", answered by asking again.
    pub category_unclear: String,
    /// The target that must be resolved before the category is consulted. A timer question
    /// ("how long is left?") reads exactly like a factual question, so this target is checked
    /// first. Empty disables the rule.
    #[serde(default)]
    pub timer_target: String,
    /// Which reply id the decision logic uses for the outcomes it produces itself, as opposed
    /// to the ones an action produces.
    pub special_replies: SpecialReplies,
    /// Confidence below which the target is ignored and the most confident action anywhere wins.
    #[serde(default = "default_target_gate")]
    pub target_gate: f64,
    /// What the provider charges per million input tokens, for the cost line in the log.
    /// Zero means "do not estimate".
    #[serde(default)]
    pub input_price_per_million: f64,
    /// The option value meaning "nothing here". Used wherever an option must be told apart from
    /// a real answer, so it can be renamed without touching the logic.
    #[serde(default = "default_none_option")]
    pub none_option: String,
    /// How each controllable thing turns an answer into an action, keyed by the target id that
    /// `target_question` may answer with.
    pub targets: BTreeMap<String, Target>,
}

fn default_none_option() -> String {
    "none".into()
}

/// One controllable target: which question names its action, and where the argument comes from.
///
/// The action id is built as `<prefix>_<option>`, which is the same shape the reply ids in
/// `[replies]` use, so adding an action to the config file is enough — provided the platform
/// driver knows the id. A target whose questions the config does not define simply never fires.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    /// Question whose answer names what to do.
    pub action_question: String,
    /// Question whose answer is the action's argument, if any.
    #[serde(default)]
    pub arg_question: Option<String>,
    /// The argument must be one of `[apps] browsers`; anything else means "no browser named".
    #[serde(default)]
    pub arg_from_browsers: bool,
    /// Question that tells two similarly named things apart — Spotify's own volume from the
    /// system's, say.
    #[serde(default)]
    pub scope_question: Option<String>,
    /// Question whose answer names the subject that `scope_option` refers to. For volume this
    /// is the "which app is named?" question, because "turn Spotify down" must choose Spotify's
    /// own volume even when the scope question itself is unsure.
    #[serde(default)]
    pub scope_subject_question: Option<String>,
    /// The option in `scope_question` that selects the alternate prefix.
    #[serde(default)]
    pub scope_option: Option<String>,
    /// Prefix used instead of the target's own name when `scope_option` is chosen.
    #[serde(default)]
    pub scope_prefix: Option<String>,
    /// Action id prefix. Defaults to the target's key.
    #[serde(default)]
    pub prefix: Option<String>,
}

fn default_target_gate() -> f64 {
    0.5
}

/// The reply ids for the outcomes the decision logic produces by itself. These are ids the
/// config defines in `[replies]`, named here so nothing about what the assistant says is
/// spelled in the code.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialReplies {
    /// The user asked a question and no language model is configured.
    pub information: String,
    /// The user was just talking.
    pub chit_chat: String,
    /// Nothing was understood — first time.
    pub clarify: String,
    /// Nothing was understood — second time in a row.
    pub give_up: String,
    /// The action is understood but this driver cannot do it.
    pub unsupported: String,
    /// Both halves of a two-part request ran.
    pub compound_done: String,
    /// A plain timer finished.
    pub timer_done: String,
    /// A reminder finished. `{label}` is what it was about.
    pub reminder_done: String,
    /// A duration could not be heard ("set a timer for ... what?").
    pub timer_unclear: String,
    /// A timer was asked about or cancelled and none is running.
    pub no_timer: String,
    /// Every timer was cancelled at once.
    pub timers_cancel_all: String,
}

/// The number vocabulary timers are parsed with, so "twenty five minutes" works in whatever
/// language the config is written for.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Timers {
    /// Word -> value, for the numbers a duration can be built from.
    pub numbers: BTreeMap<String, u32>,
    /// Word -> seconds, for the units: "minutes" -> 60.
    pub units: BTreeMap<String, u32>,
    /// How each named unit is said out loud, so "5 minutes" is not hardcoded English.
    pub spoken: Vec<SpokenUnit>,
    /// The word that introduces what a reminder is about ("remind me ... to call mum").
    pub to_word: String,
}

/// How one unit of time is spoken, singular and plural.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpokenUnit {
    /// How many seconds this unit is. Must match a value in `units`.
    pub seconds: u32,
    pub one: String,
    pub many: String,
}

/// One typed question. The shape sent over the wire differs per kind: a choice sends a map of
/// option to description, a score sends an ordered array of level descriptions.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    #[serde(rename = "type")]
    pub kind: QuestionKind,
    /// What the model should decide. `{name}` is replaced with the number of configured apps.
    pub instructions: String,
    /// Choice options: option id -> description. An empty description is sent as a null,
    /// which is what the original does for options that need no explanation.
    #[serde(default)]
    pub criteria: BTreeMap<String, String>,
    /// Score levels, ordered from lowest to highest.
    #[serde(default)]
    pub levels: Vec<String>,
    /// Merge every kind in `apps.json` into this question's options, each with no description.
    /// This is how the "which app is named?" question follows the app list instead of
    /// duplicating it — the original did the same in code, which made the two drift.
    #[serde(default)]
    pub include_apps: bool,
    /// Also offer `none`, the "nothing named here" option.
    #[serde(default)]
    pub include_none: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionKind {
    Choice,
    Noul,
    Score,
}

// --------------------------------------------------------------------------- providers

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Llm {
    /// Which implementation answers questions: `openai_compatible` today.
    pub provider: String,
    pub base_url: String,
    pub model: String,
    pub key_env: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// The system prompt. No default: how the assistant is told to speak is a choice.
    pub system_prompt: String,
    /// Provider plugins, passed through verbatim in the request.
    ///
    /// On OpenRouter this is how a router is tuned — `[{ id = "jev-router", cost_tier = "medium" }]`
    /// — and the tiers are the router's main knob, so hardcoding them would be the same mistake
    /// as hardcoding a model name. Passed through rather than modelled here because the field is
    /// provider-specific and this crate has no business knowing its schema.
    #[serde(default)]
    pub plugins: Vec<serde_json::Value>,
}

fn default_max_tokens() -> u32 {
    80
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Tts {
    /// `fish`, `openai_compatible`, `command` or `none`.
    pub provider: String,
    /// Endpoint for the HTTP providers.
    #[serde(default)]
    pub base_url: String,
    /// Voice identifier. Empty means the provider's own default.
    #[serde(default)]
    pub voice: String,
    /// The provider's model string. Empty means the provider's own default.
    #[serde(default)]
    pub model: String,
    /// Audio container requested from the provider, and the extension used in the cache.
    #[serde(default = "default_audio_format")]
    pub format: String,
    /// Name of the key, resolved by [`crate::secrets`]. Unused by `command` and `none`.
    #[serde(default)]
    pub key_env: String,
    /// Where rendered lines are cached. Relative paths are resolved against `state_dir`.
    #[serde(default = "default_tts_cache")]
    pub cache_dir: String,
    /// Shell command for `provider = "command"`. `{file}` is the rendered audio file and
    /// `{text}` the line; the command must not need a shell escape from us.
    #[serde(default)]
    pub command: Vec<String>,
}

fn default_audio_format() -> String {
    "wav".into()
}

fn default_tts_cache() -> String {
    "cache/tts".into()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Stt {
    /// `openai_compatible`, `command` or `none`.
    pub provider: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub key_env: String,
    /// Shell command for a local transcriber such as whisper.cpp. `{file}` is the audio file.
    #[serde(default)]
    pub command: Vec<String>,
    /// A prompt to bias the transcriber. Empty means none.
    #[serde(default)]
    pub prompt: Option<String>,
}

// --------------------------------------------------------------------------- keys, apps, wake

/// Maps a name used in the rest of the config to the key that holds its value. The values are
/// never stored here: look in the environment, then a `.env` beside the config, then the OS
/// keychain.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Keys {
    /// Extra `.env`-style files to read, in order. Relative paths resolve against the config.
    #[serde(default)]
    pub dotenv_files: Vec<String>,
    /// Keychain service name used on macOS, so the original's stored keys keep working.
    #[serde(default = "default_keychain_service")]
    pub keychain_service: String,
}

fn default_keychain_service() -> String {
    "com.jevsiri.keys".into()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Apps {
    /// File listing the controllable applications. Relative paths resolve against the config.
    pub file: String,
    /// Browser used for "new tab" when none is named and none is in front.
    #[serde(default)]
    pub default_browser: Option<String>,
    /// Application keys that are browsers, so "open X in <browser>" can find one.
    #[serde(default)]
    pub browsers: Vec<String>,
}

impl Default for Apps {
    fn default() -> Self {
        Self {
            file: "apps.json".into(),
            default_browser: None,
            browsers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Wake {
    /// The wake-word alternatives, joined with `|` into one pattern. Whisper mishears "Jev" as
    /// Jeff or Jeb, so the list is configuration rather than an accident of the regex.
    pub names: String,
    /// Optional full override for the wake pattern, used verbatim. Empty means build one from
    /// `names` and `greetings`.
    #[serde(default)]
    pub pattern: Option<String>,
    /// Words that may precede the name ("hey", "hi"), joined with `|`.
    #[serde(default)]
    pub greetings: Option<String>,
    /// How long the assistant stays armed after a bare wake word, in seconds.
    #[serde(default = "default_wake_window")]
    pub window_secs: f64,
    /// Sound played when the wake word is heard. Empty means silent.
    #[serde(default)]
    pub chime: Vec<String>,
    /// Pattern that stops dictation, e.g. "stop transcribing". Empty disables the phrase.
    #[serde(default)]
    pub dictation_stop: Option<String>,
    /// Pattern that starts dictation. Empty disables the phrase.
    #[serde(default)]
    pub dictation_start: Option<String>,
}

fn default_wake_window() -> f64 {
    10.0
}

impl Default for Wake {
    fn default() -> Self {
        Self {
            names: "jev".into(),
            pattern: None,
            greetings: None,
            window_secs: default_wake_window(),
            chime: Vec::new(),
            dictation_stop: None,
            dictation_start: None,
        }
    }
}

/// Recognising a website in a spoken sentence.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Web {
    /// Top-level domains that end a spoken address. A closed list, because "open the BBC"
    /// would otherwise be read as the host `bbc`.
    pub tlds: Vec<String>,
    /// The word a speaker uses for a dot ("open youtube dot com"), matched case-insensitively.
    pub dot_word: String,
    /// Prompt for the language model that resolves a site the pattern cannot ("open the BBC").
    /// Empty disables that fallback.
    #[serde(default)]
    pub prompt: String,
}

/// Scripted replies, keyed by the reply id an action or the decision logic produces. Each id
/// holds one or more alternatives picked at random. `{name}` placeholders are filled per action.
///
/// There is no built-in set: the assistant's voice is content, not code.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(transparent)]
pub struct Replies {
    pub lines: BTreeMap<String, Vec<String>>,
}

impl Replies {
    /// Pick a line for `key`, filling `{placeholder}` from `fmt`.
    ///
    /// Returns an error naming the key when the config has no line for it: an action that runs
    /// and then greets the user with silence is worse than a startup that says which key is missing.
    pub fn render(
        &self,
        key: &str,
        fmt: &BTreeMap<String, String>,
        pick: usize,
    ) -> Result<String> {
        let alternatives = self
            .lines
            .get(key)
            .with_context(|| format!("no reply configured for `{key}`"))?;
        if alternatives.is_empty() {
            bail!("reply `{key}` is configured with no lines");
        }
        let line = &alternatives[pick % alternatives.len()];
        let mut out = line.clone();
        for (name, value) in fmt {
            out = out.replace(&format!("{{{name}}}"), value);
        }
        Ok(out)
    }
}

// --------------------------------------------------------------------------- loading

impl Config {
    /// Read and validate a config file.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut config: Config = toml::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))?;
        config.source_path = Some(
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()),
        );
        config.validate()?;
        config.apps_list = AppList::load(&config.apps_file())?;
        Ok(config)
    }

    /// Read the app list without touching a config file. Used by the tests.
    pub fn load_apps(&mut self) -> Result<()> {
        self.apps_list = AppList::load(&self.apps_file())?;
        Ok(())
    }

    /// Parse from a string. Used by the tests and by `--check`.
    pub fn from_toml(text: &str) -> Result<Self> {
        let config: Config = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Catch the mistakes that would otherwise show up as strange behaviour at runtime.
    fn validate(&self) -> Result<()> {
        if self.general.gate < 0.0 || self.general.gate > 1.0 {
            bail!("general.gate must be between 0 and 1, got {}", self.general.gate);
        }
        if self.intent.questions.is_empty() {
            bail!("intent.questions is empty: the assistant has nothing to ask");
        }
        // Both decision vendors answer 400 on a missing model, and Perplexity also on an
        // unknown one. Catching it here means a config mistake is a startup error rather than a
        // failure the first time someone speaks.
        if self.intent.model.trim().is_empty() {
            bail!("intent.model is empty: every intent provider needs a model");
        }
        for (field, id) in [
            ("compound_question", &self.intent.compound_question),
            ("category_question", &self.intent.category_question),
            ("target_question", &self.intent.target_question),
        ] {
            self.require_question(id)
                .with_context(|| format!("intent.{field}"))?;
        }
        if !self.intent.timer_target.is_empty() {
            if self.timers.is_none() {
                bail!(
                    "intent.timer_target is `{}` but there is no [timers] section: timers cannot be parsed or run",
                    self.intent.timer_target
                );
            }
            for target in self.intent.targets.keys() {
                if target != &self.intent.timer_target {
                    continue;
                }
            }
        }
        for id in &self.intent.split_slots {
            self.require_question(id)?;
        }
        if self.intent.targets.is_empty() {
            bail!("intent.targets is empty: nothing can be acted on");
        }
        for (name, target) in &self.intent.targets {
            self.require_question(&target.action_question).with_context(|| {
                format!("intent.targets.{name}.action_question")
            })?;
            for (field, question) in [
                ("arg_question", &target.arg_question),
                ("scope_question", &target.scope_question),
                ("scope_subject_question", &target.scope_subject_question),
            ] {
                if let Some(question) = question {
                    self.require_question(question)
                        .with_context(|| format!("intent.targets.{name}.{field}"))?;
                }
            }
        }
        for (id, question) in &self.intent.questions {
            match question.kind {
                QuestionKind::Choice => {
                    if question.criteria.is_empty() && !question.include_apps {
                        bail!("intent.questions.{id} is a choice with no options");
                    }
                }
                QuestionKind::Score => {
                    if question.levels.len() < 2 {
                        bail!("intent.questions.{id} is a score with fewer than two levels");
                    }
                }
                QuestionKind::Noul => {}
            }
        }
        Ok(())
    }

    fn require_question(&self, id: &str) -> Result<()> {
        if !self.intent.questions.contains_key(id) {
            bail!("intent refers to question `{id}`, which is not defined in intent.questions");
        }
        Ok(())
    }

    /// Directory the state (log, cache, clips) lives in.
    pub fn state_dir(&self) -> PathBuf {
        expand(&self.general.state_dir, self.base_dir().as_deref())
    }

    /// Directory the config came from, used to resolve relative paths in it.
    pub fn base_dir(&self) -> Option<PathBuf> {
        self.source_path
            .as_ref()
            .and_then(|p| p.parent().map(Path::to_path_buf))
    }

    /// Resolve a path from the config: absolute as-is, otherwise relative to the config file.
    pub fn resolve(&self, value: &str) -> PathBuf {
        expand(value, self.base_dir().as_deref())
    }

    /// The TTS cache directory.
    pub fn tts_cache_dir(&self) -> PathBuf {
        let configured = self.resolve(&self.tts.cache_dir);
        if self.tts.cache_dir.starts_with('/') || self.tts.cache_dir.starts_with('~') {
            configured
        } else {
            self.state_dir().join(&self.tts.cache_dir)
        }
    }

    /// The app list file.
    pub fn apps_file(&self) -> PathBuf {
        self.resolve(&self.apps.file)
    }
}

/// Expand a leading `~` and join a relative path onto `base`.
pub fn expand(value: &str, base: Option<&Path>) -> PathBuf {
    let path = if let Some(rest) = value.strip_prefix("~/") {
        match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(rest),
            None => PathBuf::from(value),
        }
    } else if value == "~" {
        std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(value))
    } else {
        PathBuf::from(value)
    };
    if path.is_absolute() {
        path
    } else if let Some(base) = base {
        base.join(path)
    } else {
        path
    }
}

/// The config path to use: an explicit one, else `$HEY_JEV_CONFIG`, else `./config.toml`,
/// else `~/.config/hey-jev/config.toml`.
pub fn resolve_config_path(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    if let Some(value) = std::env::var_os("HEY_JEV_CONFIG") {
        return Some(PathBuf::from(value));
    }
    for candidate in candidate_paths() {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    candidate_paths().into_iter().next()
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut paths = vec![PathBuf::from(DEFAULT_FILENAME)];
    if let Some(home) = std::env::var_os("HOME") {
        paths.push(PathBuf::from(home).join(".config/hey-jev").join(DEFAULT_FILENAME));
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped example must parse and pass validation. If this fails, `--init` hands the
    /// user a config the program then refuses to run.
    #[test]
    fn example_config_parses() {
        let config = Config::from_toml(EXAMPLE_CONFIG).expect("example config parses");
        assert!(config.intent.questions.len() >= 10, "example should carry the full question set");
        assert!(config.general.gate > 0.0);
    }

    /// Every question id the decision logic is wired to must exist, or the assistant silently
    /// stops being able to tell a command from a question.
    #[test]
    fn example_config_wires_every_referenced_question() {
        let config = Config::from_toml(EXAMPLE_CONFIG).unwrap();
        assert!(!config.intent.compound_question.is_empty());
        assert!(!config.intent.category_question.is_empty());
        assert!(!config.intent.target_question.is_empty());
        assert!(!config.intent.split_slots.is_empty());

        // Each of the three category options must be one the category question offers, or a
        // whole branch of the decision is unreachable.
        let category = &config.intent.questions[&config.intent.category_question];
        for option in [
            &config.intent.category_information,
            &config.intent.category_chit_chat,
            &config.intent.category_unclear,
        ] {
            assert!(
                category.criteria.contains_key(option),
                "the category question has no option `{option}`"
            );
        }
    }

    /// Every target must be an option of the target question, and must name questions that
    /// exist. A target that fails either test can never be selected.
    #[test]
    fn every_target_is_reachable() {
        let config = Config::from_toml(EXAMPLE_CONFIG).unwrap();
        let options = &config.intent.questions[&config.intent.target_question].criteria;
        for (name, target) in &config.intent.targets {
            assert!(
                options.contains_key(name),
                "`{name}` is not an option of the target question"
            );
            assert!(
                config.intent.questions.contains_key(&target.action_question),
                "target `{name}` names a question that does not exist"
            );
        }
    }

    /// Every reply id the config promises must have lines, in both the actions and the ones the
    /// decision logic produces itself.
    #[test]
    fn every_promised_reply_has_lines() {
        let config = Config::from_toml(EXAMPLE_CONFIG).unwrap();
        let special = &config.intent.special_replies;
        for key in [
            &special.information,
            &special.chit_chat,
            &special.clarify,
            &special.give_up,
            &special.unsupported,
            &special.compound_done,
            &special.timer_done,
            &special.reminder_done,
            &special.timer_unclear,
            &special.no_timer,
            &special.timers_cancel_all,
        ] {
            assert!(
                config.replies.lines.contains_key(key),
                "the config names reply `{key}` but defines no lines for it"
            );
        }
        // And every action any target can produce must have a reply.
        for (name, target) in &config.intent.targets {
            let prefix = target.prefix.clone().unwrap_or_else(|| name.clone());
            let question = &config.intent.questions[&target.action_question];
            for option in question.criteria.keys() {
                let action = format!("{prefix}_{option}");
                if option == &config.intent.none_option {
                    continue;
                }
                assert!(
                    config.replies.lines.contains_key(&action),
                    "action `{action}` has no reply configured"
                );
            }
        }
    }

    /// A typo must be an error, not a setting that looks applied and is not.
    #[test]
    fn unknown_key_is_rejected() {
        assert_eq!(Config::from_toml(EXAMPLE_CONFIG).unwrap().general.mode, Mode::Ptt);

        let broken = EXAMPLE_CONFIG.replace("gate = 0.65", "gat = 0.65");
        assert!(Config::from_toml(&broken).is_err(), "a mistyped key must fail");
        assert_ne!(broken, EXAMPLE_CONFIG, "the example must actually contain the key under test");
    }

    #[test]
    fn relative_paths_resolve_against_the_config() {
        let mut config = Config::from_toml(EXAMPLE_CONFIG).unwrap();
        config.source_path = Some(PathBuf::from("/etc/hey-jev/config.toml"));
        assert_eq!(config.apps_file(), PathBuf::from("/etc/hey-jev/apps.json"));
        // The audio cache is runtime state, so a relative cache_dir lands under the state dir.
        config.general.state_dir = "/var/lib/hey-jev".into();
        assert_eq!(config.tts_cache_dir(), PathBuf::from("/var/lib/hey-jev/cache/tts"));
        // An absolute one is taken as written.
        config.tts.cache_dir = "/tmp/jev-cache".into();
        assert_eq!(config.tts_cache_dir(), PathBuf::from("/tmp/jev-cache"));
    }

    /// A level name the config does not map must say so, not silently mean 0.
    #[test]
    fn an_unmapped_volume_level_is_an_error() {
        let config = Config::from_toml(EXAMPLE_CONFIG).unwrap();
        assert_eq!(config.general.volume.level("quiet").unwrap(), 25);
        assert!(config.general.volume.level("thunderous").is_err());
    }

    #[test]
    fn tilde_expands() {
        // No set_var here: the test binary shares one environment, and mutating HOME would
        // race every other test that reads a path.
        let home = std::env::var("HOME").expect("HOME is set in the test environment");
        assert_eq!(expand("~/x", None), PathBuf::from(home).join("x"));
        assert_eq!(expand("rel", Some(Path::new("/base"))), PathBuf::from("/base/rel"));
        assert_eq!(expand("/abs", Some(Path::new("/base"))), PathBuf::from("/abs"));
    }

    #[test]
    fn reply_render_fills_placeholders_and_reports_missing_keys() {
        let mut config = Config::from_toml(EXAMPLE_CONFIG).unwrap();
        let mut fmt = BTreeMap::new();
        fmt.insert("app".to_string(), "Spotify".to_string());
        let line = config.replies.render("app_open", &fmt, 0).unwrap();
        assert!(!line.is_empty());
        let err = config.replies.render("nope", &fmt, 0).unwrap_err().to_string();
        assert!(err.contains("nope"));
        // Rendering must be deterministic for a given pick, so cached audio stays correct.
        config.replies.lines.insert("t".into(), vec!["a {x}".into(), "b {x}".into()]);
        let mut fmt = BTreeMap::new();
        fmt.insert("x".to_string(), "1".to_string());
        assert_eq!(config.replies.render("t", &fmt, 0).unwrap(), "a 1");
        assert_eq!(config.replies.render("t", &fmt, 1).unwrap(), "b 1");
    }
}
