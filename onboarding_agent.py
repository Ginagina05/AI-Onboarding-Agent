#!/usr/bin/env python3
"""
AI Onboarding Agent – Code & Business Rules Reader
=================================================

Single-file tool to scan repos, detect business rules, create a small TF-IDF index,
generate an onboarding report, and optionally ask GPT with repo context or general
questions. Uses the official OpenAI Python client; set OPENAI_API_KEY in your env.
"""
from __future__ import annotations

import argparse 
import ast
import json
import math
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

# OPTIONAL: install via `pip install openai httpx`
from openai import OpenAI
import httpx

# ---------------------------- OpenAI client ---------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Allow the script to still run for pure local operations (index/report/chat)
    # but warn the user if they try to call GPT features.
    _OPENAI_AVAILABLE = False
    client = None
else:
    _OPENAI_AVAILABLE = True
    # Use a simple httpx client (keep SSL verification on by default)
    httpx_client = httpx.Client()
    client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx_client)


# ---------------------------- Utility & Types ---------------------------- #

SUPPORTED_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".sql", ".json", ".yaml", ".yml", ".toml", ".env",
}

@dataclass
class Snippet:
    doc_id: str
    path: str
    language: str
    line_start: int
    line_end: int
    kind: str
    title: str
    text: str
    signals: List[str]


# ---------------------------- Tokenization ---------------------------- #

WORD_RE = re.compile(r"[A-Za-z0-9_\-\.]+")
STOP = set(
    """
    the a an and or of to in for with on at as is are was were be been being from by into over out
    this that these those not no yes true false null none if else elif then when while do does did done
    return import class def let const var new try catch finally public private protected static void
    extends implements package using use require module export function end begin select insert update delete
    where join left right inner outer group order limit offset create alter drop table index view materialized
    constraint primary foreign key values set get post put patch http https json xml yaml yml toml env cfg ini
    api v1 v2 v3 request response status code id ids uuid url uri ssl tls jwt oauth sql nosql mongo redis kafka
    test tests unit integration e2e mock stub spy todo fixme hack
    """.split()
)

def tokenize(text: str) -> List[str]:
    words = [w.lower() for w in WORD_RE.findall(text)]
    return [w for w in words if w not in STOP and len(w) > 1]


# ---------------------------- Scanners ---------------------------- #

BUSINESS_RULE_KEYWORDS = [
    r"rule", r"policy", r"eligib", r"qualif", r"validate", r"validation", r"approval", r"sanction",
    r"pricing", r"discount", r"surcharge", r"threshold", r"limit", r"cap", r"rate", r"quota", r"penalty",
    r"sla", r"service\s*level", r"compliance", r"gdpr", r"popia", r"hipaa", r"pci", r"sox",
    r"business\s*day", r"cutoff", r"grace\s*period", r"blackout",
]

BUSINESS_RULE_NUMBER_HINT = re.compile(r"\b(\d+[.,]?\d*)\b")

ROUTE_HINTS = [
    re.compile(r"@app\.route\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"@app\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"@router\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"path\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"app\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"router\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"@RequestMapping\(.*value\s*=\s*\{?\s*\"([^\"]+)\""),
    re.compile(r"@(Get|Post|Put|Patch|Delete)Mapping\(\s*\"([^\"]+)\""),
]

ENV_HINTS = [re.compile(r"\b[A-Z0-9_]{3,}\b\s*=\s*[^\n]+"), re.compile(r"process\.env\.[A-Z0-9_]{3,}")]

SQL_DDL_HINT = re.compile(r"\bCREATE\s+(TABLE|VIEW|INDEX)|\bALTER\s+TABLE|\bFOREIGN\s+KEY|\bPRIMARY\s+KEY", re.I)

IMPORT_HINTS = {
    ".py": re.compile(r"^\s*import\s+([\w\.]+)|^\s*from\s+([\w\.]+)\s+import", re.M),
    ".js": re.compile(r"^\s*import\s+.*from\s+['\"]([^'\"]+)['\"]|^\s*const\s+.*=\s*require\(['\"]([^'\"]+)['\"]\)", re.M),
    ".ts": re.compile(r"^\s*import\s+.*from\s+['\"]([^'\"]+)['\"]", re.M),
    ".java": re.compile(r"^\s*import\s+([\w\.]+);", re.M),
}

CONFIG_EXTS = {".json", ".yaml", ".yml", ".toml", ".env"}

def detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".py"}: return "python"
    if ext in {".js", ".jsx"}: return "javascript"
    if ext in {".ts", ".tsx"}: return "typescript"
    if ext in {".java"}: return "java"
    if ext in {".sql"}: return "sql"
    if ext in CONFIG_EXTS: return "config"
    return ext.strip(".") or "text"


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def chunk_lines(text: str, max_lines: int = 80) -> List[Tuple[int, int, str]]:
    lines = text.splitlines()
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        chunk = "\n".join(lines[start:end])
        chunks.append((start + 1, end, chunk))
        start = end
    return chunks


# ---------------------------- Business Rule Extraction ---------------------------- #

def extract_business_rules(path: Path, text: str) -> List[Snippet]:
    rules: List[Snippet] = []
    lang = detect_language(path)
    lines = text.splitlines()

    # Keyword-based sweep with number anchors
    pattern = re.compile("|".join(BUSINESS_RULE_KEYWORDS), re.I)
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            nums = BUSINESS_RULE_NUMBER_HINT.findall(line)
            title = f"Business rule hint: {line.strip()[:80]}"
            rules.append(
                Snippet(
                    doc_id=f"{path}:{i}:{i}",
                    path=str(path),
                    language=lang,
                    line_start=i,
                    line_end=i,
                    kind="rule",
                    title=title,
                    text=line.strip(),
                    signals=["business_rule"] + (["numeric_threshold"] if nums else []),
                )
            )

    # Python-specific: look into function bodies for guards/validations
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    src_lines = lines[node.lineno - 1: (node.end_lineno or node.lineno)]
                    src = "\n".join(src_lines)
                    if re.search(r"raise\s+\w+Error|assert|if\s+.+:.*return|return\s+False", src):
                        rules.append(
                            Snippet(
                                doc_id=f"{path}:{node.lineno}:{node.end_lineno}",
                                path=str(path),
                                language=lang,
                                line_start=node.lineno,
                                line_end=node.end_lineno or node.lineno,
                                kind="function_validation",
                                title=f"Validation/guard in {node.name}()",
                                text=src.strip(),
                                signals=["business_rule", "validation"],
                            )
                        )
        except Exception:
            pass

    return rules


# ---------------------------- Routes, Configs, DB, Imports ---------------------------- #

def extract_routes(path: Path, text: str) -> List[Snippet]:
    lang = detect_language(path)
    items: List[Snippet] = []
    for rx in ROUTE_HINTS:
        for m in rx.finditer(text):
            # derive route from capture groups
            route = None
            if m.lastindex:
                for i in range(1, m.lastindex + 1):
                    g = m.group(i)
                    if g and (g.startswith("/") or g.startswith("http") or "/" in g):
                        route = g
                        break
                # fallback to first non-empty group
                if not route:
                    route = next((m.group(i) for i in range(1, m.lastindex + 1) if m.group(i)), None)
            if route:
                line = text[: m.start()].count("\n") + 1
                items.append(
                    Snippet(
                        doc_id=f"{path}:{line}:{line}",
                        path=str(path),
                        language=lang,
                        line_start=line,
                        line_end=line,
                        kind="route",
                        title=f"Route: {route}",
                        text=text[m.start(): m.end()],
                        signals=["api"],
                    )
                )
    return items


def extract_configs(path: Path, text: str) -> List[Snippet]:
    items: List[Snippet] = []
    lang = detect_language(path)
    for rx in ENV_HINTS:
        for m in rx.finditer(text):
            line = text[: m.start()].count("\n") + 1
            frag = text[m.start(): m.end()].strip()
            items.append(
                Snippet(
                    doc_id=f"{path}:{line}:{line}",
                    path=str(path),
                    language=lang,
                    line_start=line,
                    line_end=line,
                    kind="config",
                    title=f"Config/Env: {frag.split('=')[0].split('.')[-1][:60]}",
                    text=frag,
                    signals=["config", "flag" if "FLAG" in frag or "ENABLE" in frag or "DISABLE" in frag else "config"],
                )
            )
    return items


def extract_db(path: Path, text: str) -> List[Snippet]:
    items: List[Snippet] = []
    lang = detect_language(path)
    if SQL_DDL_HINT.search(text):
        for (s, e, chunk) in chunk_lines(text, 80):
            if SQL_DDL_HINT.search(chunk):
                items.append(
                    Snippet(
                        doc_id=f"{path}:{s}:{e}",
                        path=str(path),
                        language=lang,
                        line_start=s,
                        line_end=e,
                        kind="ddl",
                        title=f"SQL DDL in {path.name}:{s}-{e}",
                        text=chunk.strip(),
                        signals=["db"],
                    )
                )
    if path.suffix == ".py":
        if re.search(r"Column\(|relationship\(|ForeignKey\(", text):
            for (s, e, chunk) in chunk_lines(text, 120):
                if re.search(r"Column\(|relationship\(|ForeignKey\(", chunk):
                    items.append(
                        Snippet(
                            doc_id=f"{path}:{s}:{e}",
                            path=str(path),
                            language=lang,
                            line_start=s,
                            line_end=e,
                            kind="orm_model",
                            title=f"ORM model(s) in {path.name}:{s}-{e}",
                            text=chunk.strip(),
                            signals=["db"],
                        )
                    )
    return items


def extract_imports(path: Path, text: str) -> List[Snippet]:
    items: List[Snippet] = []
    lang = detect_language(path)
    rx = IMPORT_HINTS.get(path.suffix)
    if rx:
        for m in rx.finditer(text):
            mod = next((g for g in m.groups() if g), None)
            if mod:
                line = text[: m.start()].count("\n") + 1
                items.append(
                    Snippet(
                        doc_id=f"{path}:{line}:{line}",
                        path=str(path),
                        language=lang,
                        line_start=line,
                        line_end=line,
                        kind="import",
                        title=f"Import: {mod}",
                        text=text[m.start(): m.end()],
                        signals=["import"],
                    )
                )
    return items


# ---------------------------- Indexing ---------------------------- #

def scan_file(path: Path) -> List[Snippet]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    snips: List[Snippet] = []
    snips += extract_business_rules(path, text)
    snips += extract_routes(path, text)
    snips += extract_configs(path, text)
    snips += extract_db(path, text)
    snips += extract_imports(path, text)

    for (s, e, chunk) in chunk_lines(text, 120):
        title = f"Context {path.name}:{s}-{e}"
        snips.append(
            Snippet(
                doc_id=f"{path}:{s}:{e}",
                path=str(path),
                language=detect_language(path),
                line_start=s,
                line_end=e,
                kind="context",
                title=title,
                text=chunk.strip(),
                signals=["context"],
            )
        )
    return snips


def write_index(snips: List[Snippet], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in snips:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return out


# ---------------------------- OpenAI helpers ---------------------------- #

def ask_openai_with_context(query: str, snippets: List[Snippet], model: str = "gpt-4.1-mini") -> str:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI API key not configured (OPENAI_API_KEY).")
    context = "\n\n".join(
        f"File: {s.path}:{s.line_start}-{s.line_end}\n{s.text[:1000]}" for s in snippets[:6]
    )
    prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    resp = client.responses.create(model=model, input=prompt)
    # `output_text` property may be available depending on client; fall back to parsing output.
    if getattr(resp, "output_text", None):
        return resp.output_text
    # try to extract textual parts
    out = []
    for item in getattr(resp, "output", []):
        # structure may vary; try to find text content
        cont = item.get("content")
        if isinstance(cont, list):
            for c in cont:
                if c.get("type") == "output_text":
                    out.append(c.get("text", ""))
        elif isinstance(cont, dict):
            out.append(cont.get("text", ""))
    if out:
        return "\n".join(out).strip()
    # As last resort, str(resp)
    return str(resp)


def get_embedding(text: str, model: str = "text-embedding-3-small"):
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI API key not configured (OPENAI_API_KEY).")
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


# ---------------------------- Simple TF-IDF Retrieval ---------------------------- #
@dataclass
class DocVec:
    doc_id: str
    path: str
    title: str
    text: str
    tokens: List[str]
    tf: Counter
    norm: float


def build_vectors(index_path: Path) -> Tuple[List[DocVec], Dict[str, int]]:
    docs: List[DocVec] = []
    df = Counter()
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    for row in rows:
        toks = tokenize(row["title"] + "\n" + row["text"])
        tf = Counter(toks)
        for t in tf:
            df[t] += 1
        docs.append(
            DocVec(
                doc_id=row["doc_id"],
                path=row["path"],
                title=row["title"],
                text=row["text"],
                tokens=toks,
                tf=tf,
                norm=0.0,
            )
        )
    N = len(docs)
    idf = {}
    for t, d in df.items():
        idf[t] = math.log((N + 1) / (d + 1)) + 1.0
    for dv in docs:
        s = 0.0
        for t, c in dv.tf.items():
            w = (1 + math.log(c)) * idf.get(t, 0.0)
            s += w * w
        dv.norm = math.sqrt(s) or 1.0
    return docs, idf


def score_query(query: str, docs: List[DocVec], idf: Dict[str, int], topk: int = 8):
    q_tokens = tokenize(query)
    q_tf = Counter(q_tokens)
    q_vec = {}
    for t, c in q_tf.items():
        q_vec[t] = (1 + math.log(c)) * idf.get(t, 0.0)
    q_norm = math.sqrt(sum(w * w for w in q_vec.values())) or 1.0

    scored = []
    for dv in docs:
        num = 0.0
        for t, qw in q_vec.items():
            if t in dv.tf:
                dw = (1 + math.log(dv.tf[t])) * idf.get(t, 0.0)
                num += qw * dw
        sim = num / (dv.norm * q_norm)
        if sim > 0:
            scored.append((sim, dv))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]


# ---------------------------- Report Generation ---------------------------- #

def generate_report(index_path: Path, out_dir: Path) -> Path:
    sections = defaultdict(list)
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    def add(sec: str, text: str):
        sections[sec].append(text)

    add("title", "# Onboarding Report\n")

    rule_count = sum(1 for r in rows if "business_rule" in r["signals"])
    api_count = sum(1 for r in rows if "api" in r["signals"])
    db_count = sum(1 for r in rows if "db" in r["signals"])
    cfg_count = sum(1 for r in rows if "config" in r["signals"])

    add(
        "summary",
        textwrap.dedent(
            f"""
            ## Executive Summary
            - Business rule hints: **{rule_count}**
            - API endpoints/routes: **{api_count}**
            - Database/ORM snippets: **{db_count}**
            - Config/env/flags: **{cfg_count}**
            
            This report is auto-generated. Each item below links to a file
            segment to speed up onboarding for new developers.
            """
        ).strip()
    )

    def md_link(path: str, s: int, e: int) -> str:
        return f"`{path}:{s}-{e}`"

    add("rules", "\n## Business Rules (detected)\n")
    for r in rows:
        if "business_rule" in r["signals"]:
            add("rules", f"- {r['title']} — {md_link(r['path'], r['line_start'], r['line_end'])}")

    add("api", "\n## API Surface (routes)\n")
    for r in rows:
        if "api" in r["signals"]:
            add("api", f"- {r['title']} — {md_link(r['path'], r['line_start'], r['line_end'])}")

    add("db", "\n## Data Model Hints (SQL/ORM)\n")
    for r in rows:
        if "db" in r["signals"]:
            add("db", f"- {r['title']} — {md_link(r['path'], r['line_start'], r['line_end'])}")

    add("config", "\n## Configuration & Feature Flags\n")
    for r in rows:
        if "config" in r["signals"]:
            add("config", f"- {r['title']} — {md_link(r['path'], r['line_start'], r['line_end'])}")

    add("imports", "\n## Imports & Dependencies (light sketch)\n")
    for r in rows:
        if r["kind"] == "import":
            add("imports", f"- {r['title']} — {md_link(r['path'], r['line_start'], r['line_end'])}")

    add("appendix", "\n## Appendix: Context Chunks\n")
    ctx = [r for r in rows if r["kind"] == "context"][:30]
    for r in ctx:
        add(
            "appendix",
            f"### {md_link(r['path'], r['line_start'], r['line_end'])}\n\n" +
            "```\n" + r["text"][:1000] + "\n```\n"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "onboarding_report.md"
    with out_path.open("w", encoding="utf-8") as f:
        for sec in ["title", "summary", "rules", "api", "db", "config", "imports", "appendix"]:
            for line in sections.get(sec, []):
                f.write(line + "\n")
    return out_path


# ------------------------- Chat & CLI Commands ------------------- #

def cmd_index(args):
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    snips_all: List[Snippet] = []
    for p in iter_files(root):
        rel = p.relative_to(root)
        snips = scan_file(p)
        for s in snips:
            s.path = str(rel)
        snips_all.extend(snips)
    index_path = out_dir / "index.jsonl"
    write_index(snips_all, index_path)
    print(f"Indexed {len(snips_all)} snippets -> {index_path}")


def cmd_report(args):
    index_path = Path(args.index)
    out_dir = Path(args.out)
    report = generate_report(index_path, out_dir)
    print(f"Report written: {report}")


def cmd_chat_local(args):
    index_path = Path(args.index)
    docs, idf = build_vectors(index_path)
    print("\nLocal Q&A – type your question (or 'exit')\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        hits = score_query(q, docs, idf, topk=6)
        if not hits:
            print("No matches. Try different wording.")
            continue
        print("\nTop matches:\n")
        for rank, (score, dv) in enumerate(hits, 1):
            print(f"[{rank}] {dv.title}  (score={score:.3f})\n -> {dv.doc_id}\n")
        best = hits[0][1]
        excerpt = best.text
        if len(excerpt) > 1500:
            excerpt = excerpt[:1500] + "\n... [truncated]"
        print("Details (first match):\n")
        print(textwrap.indent(excerpt, "    "))
        print("\n---\n")


def cmd_chatgpt(args):
    if not _OPENAI_AVAILABLE:
        print("OpenAI API key not configured. Set OPENAI_API_KEY to enable GPT features.")
        return
    index_file = Path(args.index)
    if not index_file.exists():
        print(f"Index file not found: {index_file}")
        return
    docs, idf = build_vectors(index_file)
    print("\nChatGPT Q&A – type your question (or 'exit')\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        hits = score_query(q, docs, idf, topk=3)
        top_snips = [h[1] for h in hits]
        try:
            answer = ask_openai_with_context(q, top_snips, model="gpt-4.1-mini")
            print("\n--- GPT Answer ---\n")
            print(answer.strip())
            print("\n------------------\n")
        except Exception as e:
            print(f"Error calling OpenAI: {e}")


def cmd_general(args):
    if not _OPENAI_AVAILABLE:
        print("OpenAI API key not configured. Set OPENAI_API_KEY to enable GPT features.")
        return
    question = args.question
    try:
        resp = client.responses.create(model=args.model or "gpt-4.1-mini", input=question)
        if getattr(resp, "output_text", None):
            print(resp.output_text)
        else:
            # fallback extraction
            print(str(resp))
    except Exception as e:
        print(f"OpenAI request failed: {e}")


# ---------------------------- CLI ---------------------------- #

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="AI Onboarding Agent – Code & Business Rules Reader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(required=True)

    p_index = sub.add_parser("index", help="Scan a repo and build an index")
    p_index.add_argument("--root", required=True, help="Path to repository root")
    p_index.add_argument("--out", required=True, help="Output directory for index and artifacts")
    p_index.set_defaults(func=cmd_index)

    p_report = sub.add_parser("report", help="Generate onboarding_report.md from an index")
    p_report.add_argument("--index", required=True, help="Path to index.jsonl")
    p_report.add_argument("--out", required=True, help="Output directory")
    p_report.set_defaults(func=cmd_report)

    p_chat = sub.add_parser("chat", help="Local Q&A over the indexed snippets")
    p_chat.add_argument("--index", required=True, help="Path to index.jsonl")
    p_chat.set_defaults(func=cmd_chat_local)

    p_chatgpt = sub.add_parser("chatgpt", help="Ask GPT with repo context (requires OPENAI_API_KEY)")
    p_chatgpt.add_argument("--index", required=True, help="Path to index.jsonl")
    p_chatgpt.set_defaults(func=cmd_chatgpt)

    p_general = sub.add_parser("general", help="Ask a general question using GPT (requires OPENAI_API_KEY)")
    p_general.add_argument("question", type=str, help="Your question")
    p_general.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model to use")
    p_general.set_defaults(func=cmd_general)

    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
