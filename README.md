# grq-cli

A command-line tool for Groq's hosted models (Llama, GPT-OSS, Qwen and others) written with only the Python standard library: no packages to install, no virtual environment. I use it next to [gem-cli](https://github.com/Ishu-dev/gem-cli) for quick text jobs, because Groq answers in a second or two and its free tier allows about 1,000 requests a day.

## Install

```
uv tool install git+https://github.com/Ishu-dev/grq-cli
mkdir -p ~/.config/grq && echo "GROQ_API_KEY=gsk_..." > ~/.config/grq/.env
```

Or skip installing: `python3 grq.py "prompt"` works on any Python 3.9+. Keys come free from console.groq.com, with no card.

## Use

```
grq "prompt"                                plain prompt
grq "summarise" -f notes.md -f data.csv     attach text files
grq "what's in this chart?" -f chart.png -m meta-llama/llama-4-scout-17b-16e-instruct   images (vision models)
cat log.txt | grq "find the error"          stdin becomes context
grq "prompt" -o out.md                      write to a file, prints token count
grq "prompt" -m openai/gpt-oss-120b         pick a model (default llama-3.3-70b-versatile, or GROQ_MODEL)
grq "prompt" -s "Answer in one line."       system prompt
grq --list                                  models with their context windows
```

### Batch mode

```
grq "Label as bug, feature or question. One word." --each "issues/*.txt" --outdir labels/ --ext .txt -j 4
```

One request per file, run in parallel, one output file per input. Files already done are skipped so a run can be resumed (`--force` redoes them). Rate limits (429) and server errors are retried, honouring Groq's `retry-after` header.

## Groq or Gemini?

| | grq-cli | gem-cli |
|---|---|---|
| Speed | Fastest | Slower |
| Inputs | Text, images | Text, images, PDF, audio, video |
| Dependencies | None | google-genai |
| Free tier | ~1,000 requests/day | Lower, varies by model |

Groq's free tier may use prompts for training, so keep personal data off it.

## Tests

```
uvx --with pytest pytest
```

The tests stub out the HTTP call, so they run offline with no key.
