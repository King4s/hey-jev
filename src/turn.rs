//! One turn: a sentence in, an action and a reply out.
//!
//! The shape is the original's. Ask everything in one call, gate on confidence, run the action,
//! then speak a pre-written line — unless the action will take the sound or the screen away, in
//! which case the line is spoken first so the assistant is not muting itself mid-sentence.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::config::Config;
use crate::intent::{decide, plan_split, split_questions, Decision, Planned};
use crate::platform::Platform;
use crate::providers::{build_intent, build_llm, build_stt, build_tts, IntentProvider, LlmProvider, SttProvider};
use crate::secrets::Secrets;
use crate::speech::Speaker;
use crate::timers::{self, Timers};
use crate::web;

/// What happened, for the log and for `--text`.
#[derive(Debug, Default, Clone)]
pub struct Turn {
    pub heard: String,
    /// The text after known mishearings were corrected, when that changed anything.
    pub corrected: Option<String>,
    /// Every answer, in the order the provider returned them.
    pub answers: Vec<(String, String, f64)>,
    /// A short description of what the fan-out meant.
    pub decision: String,
    /// The actions that ran, in order.
    pub actions: Vec<String>,
    /// The line that was spoken, if any.
    pub reply: Option<String>,
    pub provider_ms: u128,
    /// Estimated cost, from the configured per-million price. Zero when unset.
    pub cost: f64,
}

pub struct Assistant {
    pub config: Config,
    pub platform: Arc<dyn Platform>,
    intent: Box<dyn IntentProvider>,
    pub llm: Option<Box<dyn LlmProvider>>,
    speaker: Arc<Speaker>,
    /// Configured but not yet used: transcription arrives with the audio pass.
    #[allow(dead_code)]
    stt: Option<Box<dyn SttProvider>>,
    /// Running countdowns, shared with the scheduler thread.
    timers: Timers,
    /// Consecutive "say that again". Two in a row gives up rather than nagging.
    misses: u32,
}

impl Assistant {
    /// Build every provider from the config. Fails at startup rather than mid-command when a
    /// provider is named that does not exist or a key it needs is missing.
    pub fn build(config: Config, platform: Arc<dyn Platform>) -> Result<Self> {
        let mut secrets = Secrets::from_config(&config);
        let intent = build_intent(&config, &mut secrets)?;
        let llm = build_llm(&config, &mut secrets)?;
        let tts = build_tts(&config, &mut secrets)?;
        let stt = build_stt(&config, &mut secrets)?;
        let speaker = Arc::new(Speaker::new(tts, config.tts_cache_dir()));
        let timers = Timers::new();
        // The scheduler speaks on its own, so it holds what it needs rather than borrowing the
        // assistant. It starts only if timers are configured at all.
        if config.timers.is_some() && !config.intent.timer_target.is_empty() {
            spawn_scheduler(&config, timers.clone(), Arc::clone(&speaker), Arc::clone(&platform));
        }
        Ok(Self { config, platform, intent, llm, speaker, stt, timers, misses: 0 })
    }

    pub fn provider_summary(&self) -> Vec<String> {
        vec![
            format!("intent:   {}", self.intent.name()),
            format!(
                "llm:      {}",
                self.llm.as_ref().map(|llm| llm.name()).unwrap_or_else(|| "none".into())
            ),
            format!("tts:      {}", self.speaker.provider_name()),
            format!(
                "stt:      {}",
                self.stt.as_ref().map(|stt| stt.name()).unwrap_or_else(|| "none".into())
            ),
            format!("platform: {}", self.platform.name()),
        ]
    }

    /// Handle one spoken sentence.
    pub fn handle(&mut self, text: &str) -> Result<Turn> {
        self.handle_with(text, false)
    }

    /// `quiet` is for a phrase that arrived mid-sentence and may not have been addressed to the
    /// assistant: an unclear one is dropped rather than answered.
    pub fn handle_with(&mut self, text: &str, quiet: bool) -> Result<Turn> {
        let mut turn = Turn { heard: text.to_string(), ..Default::default() };

        let corrected = self.config.apps_list.fix_names(text);
        if corrected != text {
            turn.corrected = Some(corrected.clone());
        }
        let text = corrected.trim();
        if text.is_empty() {
            return Ok(turn);
        }

        let app_keys: Vec<String> = self.config.apps_list.keys();
        let started = std::time::Instant::now();
        let answers = self.intent.evaluate(text, &self.config.intent.questions, &self.config.apps_list)?;
        turn.provider_ms = started.elapsed().as_millis();
        turn.cost = self.estimate_cost();
        for (id, answer) in &answers.map {
            let value = answer
                .choice
                .clone()
                .or_else(|| answer.noul.map(|p| format!("{p:.2}")))
                .or_else(|| answer.legend.get(&(answer.score.unwrap_or(0.0).round() as i64).to_string()).cloned())
                .unwrap_or_else(|| answer.kind.clone());
            turn.answers.push((id.clone(), value, answer.confidence));
        }
        let _ = app_keys;

        let mut decision = decide(&answers, &self.config);
        turn.decision = describe(&decision);
        if let Decision::Split = decision {
            let split = self.intent.evaluate(
                text,
                &split_questions(&self.config),
                &self.config.apps_list,
            )?;
            let planned = plan_split(&split, &answers, &self.config);
            decision = if planned.is_empty() {
                Decision::Clarify
            } else {
                Decision::Actions(planned)
            };
            turn.decision = describe(&decision);
        }

        match decision {
            Decision::Clarify => {
                if quiet {
                    return Ok(turn);
                }
                self.misses += 1;
                let key = if self.misses >= 2 {
                    self.config.intent.special_replies.give_up.clone()
                } else {
                    self.config.intent.special_replies.clarify.clone()
                };
                if self.misses >= 2 {
                    self.misses = 0;
                }
                self.say(&key, &BTreeMap::new(), &mut turn)?;
            }
            Decision::Reply(key) => {
                self.misses = 0;
                self.say(&key, &BTreeMap::new(), &mut turn)?;
            }
            Decision::Llm => {
                self.misses = 0;
                let Some(llm) = self.llm.as_ref() else {
                    self.say(&self.config.intent.special_replies.information.clone(), &BTreeMap::new(), &mut turn)?;
                    return Ok(turn);
                };
                // A model that is unreachable is not a reason to stop listening: say the
                // configured "I can't answer that one" line and leave the real reason on stderr
                // for whoever is reading the log.
                match llm.answer(text) {
                    Ok(line) => {
                        self.speaker.say(&line, self.platform.as_ref())?;
                        turn.reply = Some(line);
                    }
                    Err(error) => {
                        eprintln!("  {error}");
                        self.say(
                            &self.config.intent.special_replies.information.clone(),
                            &BTreeMap::new(),
                            &mut turn,
                        )?;
                    }
                }
            }
            Decision::Actions(planned) => {
                self.misses = 0;
                self.run_actions(&planned, text, &mut turn)?;
            }
            Decision::Split => unreachable!("handled above"),
        }
        Ok(turn)
    }

    fn estimate_cost(&self) -> f64 {
        // The provider does not report token counts back through the trait, so this is the
        // configured price applied to nothing rather than a fabricated total. It stays zero
        // until the provider surfaces usage, which is honest and better than a made-up number.
        let _ = &self.config.intent.input_price_per_million;
        0.0
    }

    /// Run the planned actions and speak the matching line.
    ///
    /// Anything that takes the sound or the screen away is spoken first, or the reply would be
    /// muted or hidden by the very action it describes.
    fn run_actions(&self, planned: &[Planned], text: &str, turn: &mut Turn) -> Result<()> {
        let speak_first = planned.iter().any(|action| self.kills_the_output(action));

        if speak_first {
            let line = self.default_line(planned);
            self.speaker.say(&line, self.platform.as_ref())?;
            turn.reply = Some(line);
        }

        let mut ran = 0;
        let mut deferred_reply: Option<(String, BTreeMap<String, String>)> = None;
        for action in planned {
            let result = if action.action.starts_with("browser_") {
                self.run_browser(action, text)
                    .map(|reply| deferred_reply = Some(reply))
            } else if action.action.starts_with("timer_") {
                self.run_timer(action, text)
                    .map(|reply| deferred_reply = Some(reply))
            } else if self.platform.supports(&action.action) {
                self.platform
                    .run(&action.action, action.arg.as_deref(), &self.config)
            } else {
                anyhow::bail!("the {} driver does not implement `{}`", self.platform.name(), action.action)
            };
            match result {
                Ok(()) => {
                    ran += 1;
                    turn.actions.push(match &action.arg {
                        Some(arg) => format!("{} {arg}", action.action),
                        None => action.action.clone(),
                    });
                }
                Err(error) => eprintln!("  {} failed: {error}", action.action),
            }
        }

        if speak_first {
            return Ok(());
        }

        let line = if ran == 0 {
            self.render(&self.config.intent.special_replies.unsupported.clone(), &BTreeMap::new())?
        } else if let Some((key, fmt)) = deferred_reply {
            self.render(&key, &fmt)?
        } else {
            self.default_line(planned)
        };
        self.speaker.say(&line, self.platform.as_ref())?;
        turn.reply = Some(line);
        Ok(())
    }

    /// Whether an action must be announced before it runs.
    fn kills_the_output(&self, action: &Planned) -> bool {
        if self.config.general.speak_before.contains(&action.action) {
            return true;
        }
        // Setting the volume to the quietest level is a mute by another name.
        action.action.ends_with("volume_set")
            && action
                .arg
                .as_deref()
                .and_then(|level| self.config.general.volume.levels.get(level).copied())
                == Some(0)
    }

    /// The line for a single action, or the "both done" line for several.
    fn default_line(&self, planned: &[Planned]) -> String {
        if planned.len() == 1 {
            self.render(&planned[0].reply, &planned[0].fmt)
                .unwrap_or_else(|error| {
                    eprintln!("  {error}");
                    String::new()
                })
        } else {
            self.render(&self.config.intent.special_replies.compound_done.clone(), &BTreeMap::new())
                .unwrap_or_default()
        }
    }

    fn render(&self, key: &str, fmt: &BTreeMap<String, String>) -> Result<String> {
        let pick = rand::random::<usize>();
        self.config.replies.render(key, fmt, pick)
    }

    fn say(&self, key: &str, fmt: &BTreeMap<String, String>, turn: &mut Turn) -> Result<()> {
        let line = self.render(key, fmt)?;
        self.speaker.say(&line, self.platform.as_ref())?;
        turn.reply = Some(line);
        Ok(())
    }

    /// Open a site or a new tab. Returns the reply key and its placeholders.
    fn run_browser(&self, action: &Planned, text: &str) -> Result<(String, BTreeMap<String, String>)> {
        let browser = action.arg.as_deref();
        match action.action.as_str() {
            "browser_new_tab" => {
                self.platform.new_tab(browser, &self.config)?;
                Ok(("browser_new_tab".to_string(), BTreeMap::new()))
            }
            "browser_open_site" => {
                let url = web::find_url(text, &self.config, self.llm.as_deref())?;
                let Some(url) = url else {
                    return Ok(("browser_no_site".to_string(), BTreeMap::new()));
                };
                self.platform.open_url(browser, &url, &self.config)?;
                let mut fmt = BTreeMap::new();
                fmt.insert("site".to_string(), web::site_name(&url));
                Ok(("browser_open_site".to_string(), fmt))
            }
            other => anyhow::bail!("unknown browser action `{other}`"),
        }
    }

    /// Start, check or cancel a timer. Returns the reply key and its placeholders.
    ///
    /// The duration and the reminder text are parsed from the sentence, because only the words
    /// carry them: the intent questions say *that* a timer is wanted, not for how long.
    fn run_timer(&self, action: &Planned, text: &str) -> Result<(String, BTreeMap<String, String>)> {
        let special = &self.config.intent.special_replies;
        let Some(vocab) = self.config.timers.as_ref() else {
            anyhow::bail!("a timer action was chosen but there is no [timers] section");
        };
        match action.action.as_str() {
            "timer_set" => {
                let Some(secs) = timers::parse_duration(text, vocab) else {
                    return Ok((special.timer_unclear.clone(), BTreeMap::new()));
                };
                let label = timers::parse_reminder(text, vocab, &vocab.to_word);
                let is_reminder = label.is_some();
                self.timers.add(secs, label);
                // A reminder and a plain timer are the same countdown with different words.
                Ok((if is_reminder { "reminder_set" } else { "timer_set" }.to_string(), BTreeMap::new()))
            }
            "timer_check" => match self.timers.soonest(vocab) {
                Some((_, left)) => {
                    let mut fmt = BTreeMap::new();
                    fmt.insert("left".to_string(), timers::say_duration(left, vocab));
                    Ok(("timer_check".to_string(), fmt))
                }
                None => Ok((special.no_timer.clone(), BTreeMap::new())),
            },
            "timer_cancel" => {
                let all = text.to_lowercase().contains("all");
                if self.timers.cancel(all) == 0 {
                    return Ok((special.no_timer.clone(), BTreeMap::new()));
                }
                Ok((
                    if all { special.timers_cancel_all.clone() } else { "timer_cancel".to_string() },
                    BTreeMap::new(),
                ))
            }
            other => anyhow::bail!("unknown timer action `{other}`"),
        }
    }

    /// The app key of the browser in front, used when "new tab" names none.
    pub fn front_browser(&self) -> Option<String> {
        let front = self.platform.front_app(&self.config).ok().flatten()?;
        self.config.apps.browsers.contains(&front).then_some(front)
    }
}

fn describe(decision: &Decision) -> String {
    match decision {
        Decision::Actions(planned) => planned
            .iter()
            .map(|action| action.action.clone())
            .collect::<Vec<_>>()
            .join(" + "),
        Decision::Split => "split".to_string(),
        Decision::Reply(key) => format!("reply:{key}"),
        Decision::Llm => "llm".to_string(),
        Decision::Clarify => "clarify".to_string(),
    }
}

impl Turn {
    /// The turn as a log line.
    pub fn to_log(&self) -> String {
        let mut out = format!("heard: {:?}", self.heard);
        if let Some(corrected) = &self.corrected {
            out.push_str(&format!(" (corrected to {corrected:?})"));
        }
        out.push_str(&format!("\n  decision: {} ({}ms", self.decision, self.provider_ms));
        if self.cost > 0.0 {
            out.push_str(&format!(", ${:.6}", self.cost));
        }
        out.push(')');
        for (id, value, confidence) in &self.answers {
            out.push_str(&format!("\n  {id:24} {value:24} {confidence:.2}"));
        }
        for action in &self.actions {
            out.push_str(&format!("\n  ran: {action}"));
        }
        if let Some(reply) = &self.reply {
            out.push_str(&format!("\n  said: {reply}"));
        }
        out
    }
}

/// Wait for a configured sound file, reporting rather than failing when it is missing.
pub fn play_chime(platform: &dyn Platform, chime: &[String]) -> Result<()> {
    if chime.is_empty() {
        return Ok(());
    }
    let (program, rest) = chime.split_first().context("empty chime command")?;
    let status = std::process::Command::new(program)
        .args(rest)
        .status()
        .with_context(|| format!("playing the chime with {program}"))?;
    let _ = platform;
    if !status.success() {
        eprintln!("  the chime command failed");
    }
    Ok(())
}

/// Watch the timers and announce the ones that come due.
///
/// This is the one part of the assistant that speaks without being asked, so it owns what it
/// needs instead of borrowing the assistant: a countdown that has to be polled by an incoming
/// command is not a countdown.
fn spawn_scheduler(
    config: &Config,
    timers: Timers,
    speaker: Arc<Speaker>,
    platform: Arc<dyn Platform>,
) {
    let replies = config.replies.clone();
    let special = config.intent.special_replies.clone();
    let chime = config.wake.chime.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(250));
        for timer in timers.take_due() {
            let (key, fmt) = match &timer.label {
                Some(label) => {
                    let mut fmt = BTreeMap::new();
                    fmt.insert("label".to_string(), label.clone());
                    (special.reminder_done.clone(), fmt)
                }
                None => (special.timer_done.clone(), BTreeMap::new()),
            };
            if let Err(error) = play_chime(platform.as_ref(), &chime) {
                eprintln!("  timer chime: {error}");
            }
            match replies.render(&key, &fmt, rand::random()) {
                Ok(line) => {
                    if let Err(error) = speaker.say(&line, platform.as_ref()) {
                        eprintln!("  timer alert: {error}");
                    }
                }
                Err(error) => eprintln!("  timer alert: {error}"),
            }
        }
    });
}
