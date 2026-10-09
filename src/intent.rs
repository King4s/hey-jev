//! Turning one fan-out of typed answers into one planned action.
//!
//! This is the part of the original `siri.py` that is pure logic, and it is ported faithfully:
//! a confident answer on the question that names the target wins; otherwise the single most
//! confident action anywhere does. Two actions in one sentence are separated by asking the same
//! questions twice, scoped to "the first action" and "the second", rather than by a second model.
//!
//! Everything that decides *which* question means what is read from the config
//! (`intent.targets`, `intent.*_question`). No question id is spelled here.

use std::collections::BTreeMap;

use crate::config::{Config, Question, QuestionKind};

/// The question id prefix given to the "first action" half of a compound request.
pub const FIRST_SLOT: &str = "first";
/// The question id prefix given to the "second action" half.
pub const SECOND_SLOT: &str = "second";

/// One answer as it came back from the intent provider.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Answer {
    pub kind: String,
    /// The chosen option, for a choice.
    pub choice: Option<String>,
    /// The raw probability of yes, for a noul.
    pub noul: Option<f64>,
    /// The probability-weighted level, for a score.
    pub score: Option<f64>,
    /// Level index (as a string) mapped back to its description, for a score.
    pub legend: BTreeMap<String, String>,
    pub confidence: f64,
}

/// Every answer from one provider call, keyed by question id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Answers {
    pub map: BTreeMap<String, Answer>,
}

impl Answers {
    /// The chosen option and its confidence. A missing question reads as no answer rather than
    /// panicking: a provider that drops a question should make the assistant say "again?", not die.
    pub fn choice(&self, id: &str) -> (Option<String>, f64) {
        match self.map.get(id) {
            Some(answer) => (answer.choice.clone(), answer.confidence),
            None => (None, 0.0),
        }
    }

    /// A yes/no answer as `(value, confidence)`, where confidence is distance from a coin flip.
    pub fn noul(&self, id: &str) -> (bool, f64) {
        match self.map.get(id).and_then(|answer| answer.noul) {
            Some(probability) => (probability >= 0.5, probability.max(1.0 - probability)),
            None => (false, 0.0),
        }
    }

    /// A score answer as `(level description, confidence)`, with the level read from the legend.
    pub fn score(&self, id: &str) -> (Option<String>, f64) {
        match self.map.get(id) {
            Some(answer) => {
                let index = answer.score.unwrap_or(0.0).round() as i64;
                let level = answer.legend.get(&index.to_string()).cloned();
                (level, answer.confidence)
            }
            None => (None, 0.0),
        }
    }

    /// The same answers with `prefix_` stripped from every id, so a slot can be read as if it
    /// were an ordinary fan-out.
    pub fn strip_prefix(&self, prefix: &str) -> Answers {
        let needle = format!("{prefix}_");
        Answers {
            map: self
                .map
                .iter()
                .filter_map(|(id, answer)| {
                    id.strip_prefix(&needle).map(|rest| (rest.to_string(), answer.clone()))
                })
                .collect(),
        }
    }

    /// Re-key this fan-out under a slot prefix, which is how the split call asks its questions.
    pub fn with_prefix(&self, prefix: &str) -> Answers {
        Answers {
            map: self
                .map
                .iter()
                .map(|(id, answer)| (format!("{prefix}_{id}"), answer.clone()))
                .collect(),
        }
    }

    /// Question ids the provider returned that the config never asked about. Worth noticing:
    /// it means the provider is inventing keys.
    pub fn unknown_ids(&self, config: &Config) -> Vec<String> {
        self.map
            .keys()
            .filter(|id| !config.intent.questions.contains_key(*id))
            .cloned()
            .collect()
    }
}

/// One resolved action, with everything needed to run it and to say something afterwards.
#[derive(Debug, Clone, PartialEq)]
pub struct Planned {
    /// The action id, built as `<prefix>_<option>` — the same shape the config's reply ids use.
    pub action: String,
    /// The action's argument: an app key, a volume level, a browser key.
    pub arg: Option<String>,
    /// Reply id to speak, usually equal to `action`.
    pub reply: String,
    /// Values interpolated into the reply line.
    pub fmt: BTreeMap<String, String>,
    pub confidence: f64,
}

/// What one fan-out of answers amounts to.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// One or more actions to run, in order.
    Actions(Vec<Planned>),
    /// Two actions in one sentence; the caller re-asks with the split questions.
    Split,
    /// A scripted reply, by reply id.
    Reply(String),
    /// The user asked something; send it to the language model.
    Llm,
    /// Nothing usable — ask the user to say it again.
    Clarify,
}

/// Read a fan-out and decide what it means.
pub fn decide(answers: &Answers, config: &Config) -> Decision {
    let gate = config.general.gate;
    let (category, category_confidence) = answers.choice(&config.intent.category_question);

    // A timer question is almost always phrased as a question ("how long is left?"), so the
    // configured timer target is resolved before the category is allowed to send it to the
    // language model.
    if !config.intent.timer_target.is_empty() {
        let (target, target_confidence) = answers.choice(&config.intent.target_question);
        let (compound, compound_confidence) = answers.noul(&config.intent.compound_question);
        if target.as_deref() == Some(config.intent.timer_target.as_str())
            && target_confidence >= gate
            && !(compound && compound_confidence >= gate)
        {
            if let Some(planned) = plan_target(&config.intent.timer_target, answers, config) {
                return Decision::Actions(vec![planned]);
            }
        }
    }

    let category_is = |option: &str| category.as_deref() == Some(option);

    if category_is(&config.intent.category_chit_chat) && category_confidence >= gate {
        return Decision::Reply(config.intent.special_replies.chit_chat.clone());
    }
    if category_is(&config.intent.category_information) && category_confidence >= gate {
        return Decision::Llm;
    }
    if category_is(&config.intent.category_unclear) && category_confidence >= gate {
        return Decision::Clarify;
    }

    let (compound, compound_confidence) = answers.noul(&config.intent.compound_question);
    if compound && compound_confidence >= gate {
        return Decision::Split;
    }

    if let Some(planned) = pick_action(answers, config) {
        return Decision::Actions(vec![planned]);
    }

    if category_is(&config.intent.category_information) {
        Decision::Llm
    } else {
        Decision::Clarify
    }
}

/// The best action in this fan-out.
///
/// Trust the answer that names the target if it is reasonably sure; otherwise take the single
/// most confident action anywhere. That fallback is what makes a misread target harmless.
pub fn pick_action(answers: &Answers, config: &Config) -> Option<Planned> {
    let mut best: Option<Planned> = None;

    let (target, confidence) = answers.choice(&config.intent.target_question);
    if confidence >= config.intent.target_gate {
        if let Some(target) = target {
            best = plan_target(&target, answers, config);
        }
    }
    if best.is_some() {
        return best;
    }

    let mut candidates: Vec<Planned> = config
        .intent
        .targets
        .keys()
        .filter_map(|target| plan_target(target, answers, config))
        .collect();
    candidates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.into_iter().next()
}

/// Plan one target, or `None` when the config's questions for it are not answered confidently.
pub fn plan_target(target: &str, answers: &Answers, config: &Config) -> Option<Planned> {
    let gate = config.general.gate;
    let spec = config.intent.targets.get(target)?;
    let prefix = spec.prefix.clone().unwrap_or_else(|| target.to_string());

    let (action, action_confidence) = answers.choice(&spec.action_question);
    let action = action?;
    if action == config.intent.none_option || action_confidence < gate {
        return None;
    }

    let mut fmt = BTreeMap::new();
    let mut arg = None;

    if let Some(question) = &spec.arg_question {
        match config.intent.questions.get(question).map(|q| q.kind) {
            Some(QuestionKind::Score) => {
                let (level, _) = answers.score(question);
                fmt.insert("level".to_string(), level.clone().unwrap_or_default());
                arg = level;
            }
            _ => {
                let (value, confidence) = answers.choice(question);
                let value = value.filter(|value| value != &config.intent.none_option)?;
                if spec.arg_from_browsers {
                    // A named app that is not a browser means "no browser named", not "browser
                    // is called that".
                    if confidence >= gate && config.apps.browsers.contains(&value) {
                        arg = Some(value.clone());
                    }
                } else {
                    if confidence < gate {
                        return None;
                    }
                    if let Some(say) = config.apps_list.spoken_name(&value) {
                        fmt.insert("app".to_string(), say.to_string());
                    }
                    arg = Some(value);
                }
            }
        }
    }

    // Some actions are two things wearing one name: Spotify's own volume and the system's.
    let mut prefix = prefix;
    if let (Some(scope_question), Some(scope_option), Some(scope_prefix)) = (
        &spec.scope_question,
        &spec.scope_option,
        &spec.scope_prefix,
    ) {
        let (scope, scope_confidence) = answers.choice(scope_question);
        let scope_chosen = scope.as_deref() == Some(scope_option.as_str());
        // Naming the subject is enough on its own: "turn Spotify down" must reach Spotify's own
        // volume even when the scope question itself is unsure.
        let named_subject = match &spec.scope_subject_question {
            Some(question) => {
                let (subject, subject_confidence) = answers.choice(question);
                subject.as_deref() == Some(scope_option.as_str()) && subject_confidence >= gate
            }
            None => false,
        };
        if scope_chosen && (scope_confidence >= 0.5 || named_subject) {
            prefix = scope_prefix.clone();
        }
    }

    Some(Planned {
        action: format!("{prefix}_{action}"),
        arg,
        reply: format!("{prefix}_{action}"),
        fmt,
        confidence: action_confidence,
    })
}

/// Plan both halves of a compound request from a split fan-out.
///
/// Falls back to whatever the first fan-out was sure about when the split did not separate them.
pub fn plan_split(
    split_answers: &Answers,
    first_answers: &Answers,
    config: &Config,
) -> Vec<Planned> {
    let mut planned: Vec<Planned> = Vec::new();
    for slot in [FIRST_SLOT, SECOND_SLOT] {
        let half = split_answers.strip_prefix(slot);
        if let Some(action) = pick_action(&half, config) {
            let duplicate = planned
                .iter()
                .any(|existing| existing.action == action.action && existing.arg == action.arg);
            if !duplicate {
                planned.push(action);
            }
        }
    }
    if planned.len() < 2 {
        return config
            .intent
            .targets
            .keys()
            .filter_map(|target| plan_target(target, first_answers, config))
            .collect();
    }
    planned
}

/// The questions for the split call: the same questions, asked once per slot.
pub fn split_questions(config: &Config) -> BTreeMap<String, Question> {
    let mut out = BTreeMap::new();
    for (slot, word) in [(FIRST_SLOT, "FIRST"), (SECOND_SLOT, "SECOND")] {
        for (id, question) in &config.intent.questions {
            if config.intent.compound_question == *id || config.intent.category_question == *id {
                continue;
            }
            let mut scoped = question.clone();
            scoped.instructions =
                format!("Considering ONLY the {word} action the user asks for: {}", question.instructions);
            out.insert(format!("{slot}_{id}"), scoped);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EXAMPLE_CONFIG;

    fn config() -> Config {
        let mut config = Config::from_toml(EXAMPLE_CONFIG).unwrap();
        // Tests run without an apps.json beside them; use the example app list.
        config.apps_list.say = BTreeMap::from([
            ("spotify".to_string(), "Spotify".to_string()),
            ("chrome".to_string(), "Google Chrome".to_string()),
            ("brave".to_string(), "Brave Browser".to_string()),
            ("safari".to_string(), "Safari".to_string()),
            ("slack".to_string(), "Slack".to_string()),
        ]);
        config.apps_list.names = config.apps_list.say.clone();
        config
    }

    fn choice(value: &str, confidence: f64) -> Answer {
        Answer {
            kind: "choice".into(),
            choice: Some(value.into()),
            confidence,
            ..Default::default()
        }
    }

    fn noul(value: bool) -> Answer {
        Answer {
            kind: "noul".into(),
            noul: Some(if value { 0.95 } else { 0.05 }),
            confidence: 0.95,
            ..Default::default()
        }
    }

    /// The plain case: a confident app and a confident app action become one action.
    #[test]
    fn a_confident_command_becomes_one_action() {
        let config = config();
        let answers = Answers {
            map: BTreeMap::from([
                ("category".into(), choice("mac_command", 0.99)),
                ("compound".into(), noul(false)),
                ("target".into(), choice("app", 0.9)),
                ("app".into(), choice("spotify", 0.95)),
                ("app_action".into(), choice("open", 0.9)),
            ]),
        };
        let Decision::Actions(planned) = decide(&answers, &config) else {
            panic!("expected an action, got {:?}", decide(&answers, &config));
        };
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].action, "app_open");
        assert_eq!(planned[0].arg.as_deref(), Some("spotify"));
        assert_eq!(planned[0].fmt.get("app").map(String::as_str), Some("Spotify"));
    }

    /// Below the gate nothing may be executed: the whole point of the threshold.
    #[test]
    fn a_low_confidence_action_is_not_taken() {
        let config = config();
        let answers = Answers {
            map: BTreeMap::from([
                ("category".into(), choice("mac_command", 0.99)),
                ("target".into(), choice("app", 0.9)),
                ("app".into(), choice("spotify", 0.2)),
                ("app_action".into(), choice("open", 0.9)),
            ]),
        };
        assert_ne!(decide(&answers, &config), Decision::Actions(vec![]));
        assert!(pick_action(&answers, &config).is_none());
    }

    /// When the target is unsure, the most confident action anywhere still gets taken.
    #[test]
    fn an_unsure_target_falls_back_to_the_most_confident_action() {
        let config = config();
        let answers = Answers {
            map: BTreeMap::from([
                ("category".into(), choice("mac_command", 0.99)),
                ("target".into(), choice("app", 0.4)),
                ("media_action".into(), choice("pause", 0.97)),
                ("app_action".into(), choice("none", 0.9)),
            ]),
        };
        let planned = pick_action(&answers, &config).expect("fallback action");
        assert_eq!(planned.action, "media_pause");
    }

    /// "turn Spotify down" must hit Spotify's own volume, not the system's.
    #[test]
    fn named_spotify_selects_the_spotify_volume_prefix() {
        let config = config();
        let answers = Answers {
            map: BTreeMap::from([
                ("category".into(), choice("mac_command", 0.99)),
                ("target".into(), choice("volume", 0.9)),
                ("volume_action".into(), choice("down", 0.9)),
                ("volume_scope".into(), choice("spotify", 0.3)),
                ("app".into(), choice("spotify", 0.9)),
            ]),
        };
        let planned = pick_action(&answers, &config).expect("an action");
        assert_eq!(planned.action, "spotify_volume_down");
    }

    /// A volume level is a score, read back through its legend.
    #[test]
    fn a_score_answer_is_read_through_its_legend() {
        let config = config();
        let answers = Answers {
            map: BTreeMap::from([
                ("category".into(), choice("mac_command", 0.99)),
                ("target".into(), choice("volume", 0.9)),
                ("volume_action".into(), choice("set", 0.9)),
                ("volume_level".into(), Answer {
                    kind: "score".into(),
                    score: Some(1.0),
                    legend: BTreeMap::from([
                        ("0".into(), "silent".into()),
                        ("1".into(), "quiet".into()),
                        ("2".into(), "medium".into()),
                    ]),
                    confidence: 0.9,
                    ..Default::default()
                }),
            ]),
        };
        let planned = pick_action(&answers, &config).expect("an action");
        assert_eq!(planned.action, "volume_set");
        assert_eq!(planned.arg.as_deref(), Some("quiet"));
        assert_eq!(planned.fmt.get("level").map(String::as_str), Some("quiet"));
    }

    /// A question goes to the language model, chit-chat gets a scripted reply, and gibberish
    /// asks again — the three categories must not be confused with a command.
    #[test]
    fn categories_route_to_the_right_place() {
        let config = config();
        let of = |category: &str| Answers {
            map: BTreeMap::from([
                ("category".into(), choice(category, 0.99)),
                ("compound".into(), noul(false)),
                ("target".into(), choice("none", 0.9)),
            ]),
        };
        assert_eq!(decide(&of("information_request"), &config), Decision::Llm);
        assert_eq!(decide(&of("chit_chat"), &config), Decision::Reply("chit_chat".into()));
        assert_eq!(decide(&of("unclear"), &config), Decision::Clarify);
    }

    /// A compound request is separated by re-asking, and identical halves collapse to one.
    #[test]
    fn a_compound_request_splits_into_two_actions() {
        let config = config();
        let split = Answers {
            map: BTreeMap::from([
                ("first_media_action".into(), choice("pause", 0.95)),
                ("second_app_action".into(), choice("open", 0.95)),
                ("second_app".into(), choice("slack", 0.95)),
            ]),
        };
        let planned = plan_split(&split, &Answers::default(), &config);
        assert_eq!(planned.len(), 2);
        assert_eq!(planned[0].action, "media_pause");
        assert_eq!(planned[1].action, "app_open");
        assert_eq!(planned[1].arg.as_deref(), Some("slack"));

        // A split that returns the same action twice is one action, not two.
        let same = Answers {
            map: BTreeMap::from([
                ("first_media_action".into(), choice("pause", 0.95)),
                ("second_media_action".into(), choice("pause", 0.95)),
            ]),
        };
        let mut first = Answers::default();
        first.map.insert("media_action".into(), choice("pause", 0.95));
        let planned = plan_split(&same, &first, &config);
        assert_eq!(planned.len(), 1, "duplicates must collapse");
    }

    /// "how long is left?" reads like a question but is a timer command.
    #[test]
    fn a_timer_question_beats_the_information_category() {
        let config = config();
        let answers = Answers {
            map: BTreeMap::from([
                ("category".into(), choice("information_request", 0.99)),
                ("compound".into(), noul(false)),
                ("target".into(), choice("timer", 0.95)),
                ("timer_action".into(), choice("check", 0.95)),
            ]),
        };
        let Decision::Actions(planned) = decide(&answers, &config) else {
            panic!("expected a timer action");
        };
        assert_eq!(planned[0].action, "timer_check");
    }

    /// The split call must ask the same questions twice, minus the two that describe the whole
    /// request rather than one half of it.
    #[test]
    fn split_questions_scope_each_slot() {
        let config = config();
        let questions = split_questions(&config);
        assert!(questions.contains_key("first_app_action"));
        assert!(questions.contains_key("second_app_action"));
        assert!(!questions.contains_key("first_category"));
        assert!(!questions.contains_key("second_compound"));
        assert!(questions["first_app_action"].instructions.starts_with("Considering ONLY the FIRST"));
    }
}
