//! hey-jev: say a thing, it does it and answers back.
//!
//! Every provider, model, prompt, threshold, path and reply comes from a config file. See
//! `config.example.toml`, which `--init` writes out.

mod apps;
mod config;
mod intent;
mod platform;
mod providers;
mod secrets;
mod speech;
mod timers;
mod turn;
mod web;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::Parser;

use config::Config;
use platform::{build as build_platform, Platform, RecordingPlatform};
use turn::{play_chime, Assistant};

#[derive(Parser, Debug)]
#[command(
    name = "hey-jev",
    about = "A voice assistant whose every setting lives in a config file.",
    version
)]
struct Cli {
    /// Config file. Defaults to $HEY_JEV_CONFIG, ./config.toml, then ~/.config/hey-jev/config.toml.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Run one turn on this text instead of listening.
    #[arg(short, long)]
    text: Option<String>,

    /// Resolve the action and say what would run, without running it.
    #[arg(long)]
    dry_run: bool,

    /// Write a config file with every setting and its default, then exit.
    #[arg(long)]
    init: bool,

    /// Check the config and every provider it names, then exit.
    #[arg(long)]
    check: bool,

    /// List the actions the selected platform driver implements.
    #[arg(long)]
    actions: bool,

    /// Print the resolved configuration, including where each path points.
    #[arg(long)]
    print_config: bool,

    /// Pre-render every fixed reply into the audio cache, then exit.
    #[arg(long)]
    warm: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    if cli.init {
        return init_config(cli.config.as_deref());
    }

    let path = config::resolve_config_path(cli.config.as_deref())
        .context("no config path could be determined (is $HOME set?)")?;
    if !path.is_file() {
        bail!(
            "no config at {}\nrun `hey-jev --init` to write one",
            path.display()
        );
    }
    let config = Config::load(&path)?;
    let platform = build_platform(None, &config)?;
    println!("config:   {}", path.display());
    println!("apps:     {} ({} apps)", config.apps_file().display(), config.apps_list.names.len());

    if cli.print_config {
        print_config(&config);
        return Ok(());
    }

    if cli.actions {
        let mut implemented: Vec<&str> = Vec::new();
        let mut missing: Vec<&str> = Vec::new();
        for action in platform::ALL_ACTIONS {
            if platform.supports(action) {
                implemented.push(action);
            } else {
                missing.push(action);
            }
        }
        println!("\n{} driver implements {} of {} actions:", platform.name(), implemented.len(), platform::ALL_ACTIONS.len());
        for action in &implemented {
            println!("  {action}");
        }
        if !missing.is_empty() {
            println!("  not implemented here: {}", missing.join(", "));
        }
        return Ok(());
    }

    // A dry run swaps in a driver that records instead of acting, and silences the voice: the
    // point is to see what the assistant decided, without spending anything or making a sound.
    let config = if cli.dry_run {
        let mut config = config;
        config.tts.provider = "none".to_string();
        config
    } else {
        config
    };
    let platform: Arc<dyn Platform> = if cli.dry_run {
        Arc::new(RecordingPlatform::default())
    } else {
        Arc::from(platform)
    };
    let assistant = Assistant::build(config, platform)?;

    if cli.check {
        println!("\nproviders:");
        for line in assistant.provider_summary() {
            println!("  {line}");
        }
        println!("\ncheck: the config parses, the app list loads, and every provider builds");
        return Ok(());
    }

    if cli.warm {
        let app_names: Vec<&str> = assistant
            .config
            .apps_list
            .say
            .values()
            .map(String::as_str)
            .collect();
        let levels: Vec<&str> = assistant
            .config
            .general
            .volume
            .levels
            .keys()
            .map(String::as_str)
            .collect();
        let lines = speech::expand_lines(&assistant.config.replies, &app_names, &levels);
        let speaker = speech::Speaker::new(
            providers::build_tts(&assistant.config, &mut secrets::Secrets::from_config(&assistant.config))?,
            assistant.config.tts_cache_dir(),
        );
        let rendered = speaker.warm(&lines)?;
        println!("\nrendered {rendered} new lines into {}", assistant.config.tts_cache_dir().display());
        return Ok(());
    }

    if let Some(text) = cli.text {
        let mut assistant = assistant;
        let turn = assistant.handle(&text)?;
        println!();
        println!("{}", turn.to_log());
        if cli.dry_run {
            // Nothing was executed, so the caller must be told what would have been.
            println!("\n(dry run: no action was executed, no audio was played)");
        }
        return Ok(());
    }

    // Listening needs the audio pass, which is not implemented yet. Say so plainly rather than
    // starting something that cannot hear.
    let assistant = assistant;
    if !assistant.config.wake.chime.is_empty() {
        let _ = play_chime(assistant.platform.as_ref(), &assistant.config.wake.chime);
    }
    bail!(
        "there is no microphone loop yet: this build runs one turn at a time with `--text`.\n\
         The audio capture, the wake word and the dictation bubble are the next pass."
    )
}

fn init_config(explicit: Option<&std::path::Path>) -> Result<()> {
    let path = match explicit {
        Some(path) => path.to_path_buf(),
        None => {
            let home = std::env::var_os("HOME").context("$HOME is not set")?;
            PathBuf::from(home).join(".config/hey-jev").join(config::DEFAULT_FILENAME)
        }
    };
    if path.exists() {
        bail!("{} already exists; move it aside first", path.display());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&path, config::EXAMPLE_CONFIG)
        .with_context(|| format!("writing {}", path.display()))?;
    println!("wrote {}", path.display());
    println!("the app list it refers to is apps.json, next to it; the shipped one is in this repo");
    Ok(())
}

fn print_config(config: &Config) {
    println!("\nresolved:");
    println!("  state dir:  {}", config.state_dir().display());
    println!("  audio cache:{}", config.tts_cache_dir().display());
    println!("  platform:   {}", config.general.platform.clone().unwrap_or_else(|| "auto".into()));
    println!("  mode:       {:?}", config.general.mode);
    println!("  gate:       {}", config.general.gate);
    println!("  questions:  {} ({} targets with scope questions)",
        config.intent.questions.len(),
        config.intent.targets.values().filter(|t| t.scope_question.is_some()).count()
    );
    println!("  replies:    {} keys", config.replies.lines.len());
    println!("  apps:       {}", config.apps_list.names.len());
}
