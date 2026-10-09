# hey-jev (Rust)

A voice assistant for your machine. Say "Hey Jev" or hold right Option, say a thing, it does it
and answers back.

This is a fork of [henryklunaris/hey-jev](https://github.com/henryklunaris/hey-jev), rewritten in
Rust. The original was macOS-only and carried its settings as constants at the top of a Python
file. This one has **no hardcoded settings**: which model decides, which voice speaks, which
transcriber listens, what it says, what it can control and how confident it has to be are all read
from one config file. It also runs on Linux as well as macOS.

Everything in this repository is in English.

## Status

This is the first pass, and it is honest about its edges.

**Working:** the config layer, the provider layer, the decision logic, the action table for both
macOS and Linux, timers and reminders, website recognition, reply rendering and caching, and a
headless one-turn-at-a-time CLI.

**Not built yet:** microphone capture, the wake word, push-to-talk, dictation, and the native
window. Until those land the assistant runs one turn at a time from the command line, and `--text`
is how you exercise everything that does work.

## Quick start

```bash
cargo build --release
./target/release/hey-jev --init          # writes ~/.config/hey-jev/config.toml
$EDITOR ~/.config/hey-jev/config.toml    # set your keys' names and your voice
./target/release/hey-jev --check         # config parses, every provider builds
./target/release/hey-jev --text "open spotify" --dry-run
```

`--dry-run` resolves everything and executes nothing: it swaps in a recording driver and silences
the voice, so it proves the whole pipeline on a machine that has none of the applications involved
and spends nothing.

```
$ hey-jev --text "open spotify and pause the music" --dry-run
heard: "open spotify and pause the music"
  decision: app_open + media_pause (417ms)
  ran: app_open spotify
  ran: media_pause
  said: [chuckling] Done, both of them.
```

## Configuration

Everything is in one file. `hey-jev --init` writes it with every key and a comment. There is no
other place a setting lives, and a mistyped key is an error at startup rather than a setting that
silently does nothing.

| Section | What it decides |
| --- | --- |
| `[general]` | Mode, the confidence gate, where state lives, which actions get announced before they run, what each named volume level means |
| `[intent]` | Which provider decides what a sentence means, the model, the questions it asks, and how an answer becomes an action |
| `[llm]` | Which model answers a question the user asks |
| `[tts]` | Which service speaks, with which voice and model |
| `[stt]` | Which transcriber turns speech into text |
| `[keys]` | Where to look for keys — environment, `.env`, then the OS keychain. Values are never stored in the config |
| `[apps]` | The controllable applications and which of them are browsers |
| `[wake]` | The wake words, the chimes and the dictation phrases |
| `[web]` | How a spoken sentence becomes a website |
| `[timers]` | The number vocabulary durations are parsed with |
| `[replies]` | Everything it says |

### Choosing a model

The intent provider and the language model are separate settings, because they are separate jobs.

`[intent] provider` is one of:

- `typesafe` — TypeSafe's System One endpoint returns calibrated probabilities directly. This is
  what the config ships with, and what the numbers below were measured against.
- `decider` — Perplexity's Decisions API. A **different vendor's answer to the same problem**:
  `pplx-decider-v1.1-27b` also returns probabilities instead of text, over the same
  `noul`/`choice`/`score` primitives. It speaks the same wire protocol as System One, so switching
  is the endpoint, the model name and the key — nothing else. `perplexity` is accepted as a
  spelling of `decider`. It is roughly half the price per input token ($0.02/M against $0.042/M)
  and caps at 128 questions per request and 10 requests per second.
- `openai_compatible` — any `/chat/completions` endpoint: OpenRouter, OpenAI, a local LM Studio or
  Ollama, or your own gateway. The model is asked for the same probability distribution and its
  answer is **verified**: a choice must be one of the options offered, the probabilities must sum
  to 1, and every option must appear. A fabricated distribution is rejected rather than acted on,
  because the code downstream of it launches applications.

The first two are decision models and the third is not, which is the distinction that matters:
a decision model's numbers are calibrated by training, a chat model's are a claim that has to be
checked. Running either is a config change, so they can be measured against each other on the
same sentences.

`[llm]` takes the same `openai_compatible` shape, so pointing it at a different service is three
lines.

## Providers

| Job | Implementations |
| --- | --- |
| Intent | `typesafe` (TypeSafe System One), `decider` (Perplexity Decisions), `openai_compatible` |
| Language model | `openai_compatible` |
| Speech | `fish`, `openai_compatible` (`/audio/speech`), `command` (any local synthesizer), `none` |
| Transcription | `openai_compatible`, `command` (whisper.cpp and friends, keeps audio local), `none` |

Adding one means adding a variant to the trait's match; nothing else in the crate needs to know.

## Platforms

| | macOS | Linux |
| --- | --- | --- |
| App control | AppleScript | `gtk-launch`, `xdotool`, `pkill` |
| Volume | `osascript` | `pactl` |
| Media | Spotify via AppleScript | `playerctl` |
| Dark mode | System Events | `gsettings` |
| Lock / sleep | AppleScript / `pmset` | `loginctl` / `systemctl` |
| New browser tab | yes | **not supported** |
| Audio playback | `afplay` | `paplay` |
| Clipboard | `pbcopy` | `wl-copy` or `xclip` |

A driver reports what it cannot do, and the assistant answers "I know what you want, I just can't
do that one yet" rather than failing silently. `hey-jev --actions` prints the list.

The macOS driver keeps the original's behaviour, including the two things that look like bugs: it
keeps asking Spotify to play because Spotify ignores play while it is loading, and it presses
"previous" twice unless the song has just started, because one press only restarts the track.

## How a turn works

1. Known mishearings are fixed up from the app list before anything else sees the text, so a
   provider can pick an option the transcriber never wrote.
2. One call asks every question at once — category, whether it holds two actions, what is being
   controlled, and the action for each thing that could be controlled. Extra questions cost
   tokens, not time.
3. Answers below the confidence gate are refused.
4. Two actions in one sentence are separated by asking the same questions twice, scoped to "the
   first action" and "the second" — no second model, and no LLM call to split a sentence.
5. The action runs through the platform driver.
6. A pre-written line matching the action is spoken. Lines are rendered once and cached under a
   hash of the provider, the voice and the text.
7. Anything that takes the sound or the screen away is announced *before* it runs, or the
   assistant would be muting itself mid-sentence.

## Measured

Every number here comes from a real run against the live TypeSafe API on a Linux host, with the
shipped config and no key for the language model. `cargo test`: **49 passed**.

| Said | Decided | Latency |
| --- | --- | --- |
| "open spotify" | `app_open spotify` | 397 ms |
| "turn spotify down" | `spotify_volume_down` | 642 ms |
| "open spotify and pause the music" | `app_open` + `media_pause` | 417 ms |
| "set a timer for 5 minutes" | `timer_set` | 456 ms |
| "how long is left?" | `timer_check` | 251 ms |
| "mute" | `volume_mute` | 290 ms |
| "what year was Hamlet written" | language model | 296 ms |

"turn spotify down" reaching `spotify_volume_down` rather than the system volume is the scope rule
working: naming the subject settles it even when the scope question is unsure. "how long is left?"
is answered as a timer command rather than a question, which is why the timer target is resolved
before the category.

With no key for the language model the question is answered with the configured "I can't answer
that one yet" line, and the real reason goes to stderr — a model that is unreachable is not a
reason to stop listening.

## Building and testing

The build host needs a Rust toolchain and a C linker. `scripts/check.sh` syncs the sources to the
build host, runs `cargo test`, and builds the release binary:

```bash
./scripts/check.sh
```

Override the host with `HEY_JEV_BUILD_HOST`, `HEY_JEV_BUILD_JUMP` or `HEY_JEV_BUILD_DIR`.

## What it costs

Whatever your providers charge. The config asks each provider for a price per million tokens so
the log can estimate; until a provider surfaces token counts back through the trait, that estimate
stays at zero rather than being a guess.

## Credits

The design, the question set, the reply lines and the action table come from
[henryklunaris/hey-jev](https://github.com/henryklunaris/hey-jev), which is MIT licensed.
