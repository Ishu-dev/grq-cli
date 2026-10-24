#!/usr/bin/env python3
"""Thin Groq CLI for offloading cheap tasks from Claude Code.

Usage:
  grq "prompt"                          plain prompt
  grq "prompt" -f notes.md -f data.csv  attach text files
  grq "prompt" -f shot.png              attach image (vision models only)
  cat notes.md | grq "summarise this"   stdin becomes context
  grq "prompt" -m openai/gpt-oss-120b   pick model
  grq "prompt" -o out.md                write result to file
  grq "tag as bug/feature/question" --each "issues/*.txt" --outdir tags/   one call per file
  grq --list                            list available models
"""
import argparse, base64, glob, json, mimetypes, os, pathlib, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
CONFIG_ENV = pathlib.Path.home() / ".config" / "grq" / ".env"
BASE = "https://api.groq.com/openai/v1"
TEXT_EXT = {".txt", ".md", ".csv", ".json", ".py", ".html", ".xml", ".yaml", ".yml",
            ".tex", ".srt", ".js", ".ts", ".sh", ".sql", ".toml", ".ini", ".log"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
RETRY_CODES = {429, 500, 502, 503}


class GroqError(Exception):
    pass


def load_env():
    # A .env beside the script (dev checkout), then ~/.config/grq/.env (installed tool).
    # Values already in the environment win.
    for env in (HERE / ".env", CONFIG_ENV):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call(path, key, payload=None, retries=4):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST" if payload else "GET",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in RETRY_CODES and attempt < retries:
                # Groq sends retry-after on 429; fall back to exponential backoff
                wait = float(e.headers.get("retry-after") or 2 ** attempt) if e.headers else 2 ** attempt
                print(f"grq: {e.code}, retrying in {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            body = e.read().decode(errors="replace")
            try:
                msg = json.loads(body)["error"]["message"]
            except Exception:
                msg = body[:500]
            raise GroqError(f"groq error {e.code}: {msg}")
        except urllib.error.URLError as e:
            raise GroqError(f"network error: {e.reason}")


def read_files(files):
    text_parts, image_parts = [], []
    for f in files:
        p = pathlib.Path(f).expanduser()
        if not p.exists():
            raise GroqError(f"no such file: {p}")
        ext = p.suffix.lower()
        if ext in IMG_EXT:
            mime = mimetypes.guess_type(p.name)[0] or "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode()
            image_parts.append({"type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"}})
        elif ext in TEXT_EXT or p.stat().st_size < 2_000_000:
            text_parts.append(f"--- {p.name} ---\n{p.read_text(errors='replace')}\n")
        else:
            raise GroqError(f"unsupported file type for groq: {p.name} (text and images only)")
    return text_parts, image_parts


def complete(key, a, text_parts, image_parts):
    body = "\n".join(text_parts + [a.prompt])
    content = [{"type": "text", "text": body}] + image_parts if image_parts else body
    messages = ([{"role": "system", "content": a.system}] if a.system else []) + \
               [{"role": "user", "content": content}]
    resp = call("/chat/completions", key, {
        "model": a.model,
        "messages": messages,
        "temperature": a.temperature,
        "max_tokens": a.max_tokens,
    })
    return resp["choices"][0]["message"]["content"] or "", resp.get("usage", {})


def out_path(src, outdir, ext):
    return pathlib.Path(outdir).expanduser() / (src.stem + (ext or src.suffix or ".md"))


def expand(patterns):
    files = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.expanduser(pat), recursive=True))
        files += [pathlib.Path(h) for h in hits if os.path.isfile(h)]
    return list(dict.fromkeys(files))


def run_batch(key, a, shared_text, shared_images):
    files = expand(a.each)
    if not files:
        raise GroqError(f"--each matched no files: {' '.join(a.each)}")
    outdir = pathlib.Path(a.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if a.force or not out_path(f, outdir, a.ext).exists()]
    if len(todo) < len(files):
        print(f"grq: skipping {len(files) - len(todo)} already done (use --force to redo)", file=sys.stderr)

    def one(src):
        text, images = read_files([src])
        out, _ = complete(key, a, shared_text + text, shared_images + images)
        dest = out_path(src, outdir, a.ext)
        dest.write_text(out if out.endswith("\n") else out + "\n")
        return src, dest

    failed = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(one, f): f for f in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                src, dest = fut.result()
                print(f"[{i}/{len(todo)}] {src.name} -> {dest}")
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(todo)}] {futures[fut].name} FAILED: {e}", file=sys.stderr)
    if failed:
        raise GroqError(f"{failed} of {len(todo)} files failed")


def build_parser():
    ap = argparse.ArgumentParser(prog="grq", description="Groq CLI")
    ap.add_argument("prompt", nargs="?", help="instruction for the model")
    ap.add_argument("-f", "--file", action="append", default=[], help="attach a file (repeatable)")
    ap.add_argument("-m", "--model", default=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    ap.add_argument("-o", "--out", help="write output to this path")
    ap.add_argument("-s", "--system", help="system instruction")
    ap.add_argument("-t", "--temperature", type=float, default=0.2)
    ap.add_argument("-n", "--max-tokens", type=int, default=8192)
    ap.add_argument("--list", action="store_true", help="list models and exit")
    b = ap.add_argument_group("batch mode")
    b.add_argument("--each", nargs="+", metavar="GLOB", help="run the prompt once per matching file")
    b.add_argument("--outdir", help="where batch results go (one file per input)")
    b.add_argument("--ext", help="extension for batch outputs, e.g. .md (default: same as input)")
    b.add_argument("-j", "--jobs", type=int, default=4, help="parallel requests in batch mode (default 4)")
    b.add_argument("--force", action="store_true", help="redo files that already have an output")
    return ap


def run(argv=None):
    load_env()
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.each and not a.outdir:
        ap.error("--each needs --outdir")
    if a.each and a.out:
        ap.error("use --outdir with --each, not -o")

    key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API")
    if not key:
        raise GroqError(f"GROQ_API_KEY not found. Set it in the environment or in {CONFIG_ENV}")

    if a.list:
        for m in sorted(call("/models", key)["data"], key=lambda x: x["id"]):
            print(f"{m['id']:<45} {m.get('context_window', '')}")
        return

    if not a.prompt:
        ap.error("prompt required")

    text_parts, image_parts = read_files(a.file)
    if not sys.stdin.isatty():
        stdin = sys.stdin.read()
        if stdin.strip():
            text_parts.append(f"--- stdin ---\n{stdin}\n")

    if a.each:
        run_batch(key, a, text_parts, image_parts)
        return

    text, usage = complete(key, a, text_parts, image_parts)
    if a.out:
        pathlib.Path(a.out).expanduser().write_text(text)
        print(f"wrote {a.out} ({len(text)} chars, {usage.get('total_tokens', '?')} tokens)")
    else:
        print(text)


def main():
    try:
        run()
    except GroqError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
