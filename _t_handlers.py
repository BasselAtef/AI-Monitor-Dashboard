"""Regression guard for inline onclick handlers in index.html.

Covers the class of bug behind "My Account does nothing": commit b93766d removed
`const panel = ...` from showAccountPanel() while two lines still referenced
`panel`. The function is `async`, so the ReferenceError became a rejected promise
instead of a visible throw, and the button failed silently.

Approach: pull the real JavaScript out of the template first, strip strings and
comments, then check identifiers against declared names. An earlier version of
this test matched prose inside string literals ('it', 'now', 'table') and
produced 50 false positives, so tokenisation has to happen before analysis.

No third-party parser is available offline, so string and comment stripping is
done with a small state machine.
"""
import io
import re

TPL = "templates/index.html"
tpl = io.open(TPL, encoding="utf-8").read()

fails = []


def check(label, cond, extra=""):
    print(f"  [{'ok' if cond else 'RED'}] {label}" + (f"  {extra}" if extra else ""))
    if not cond:
        fails.append(label)


def strip_js(src):
    """Blank out string literals, template literals and comments.

    Leaves offsets and line breaks intact so line numbers still line up.
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch in "\"'":
            quote = ch
            i += 1
            while i < n and src[i] != quote:
                if src[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n:
                    out[i] = " "
                    i += 1
            if i < n:
                out[i] = " "
                i += 1
            continue
        if ch == "`":
            i += 1
            while i < n and src[i] != "`":
                if src[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n:
                    out[i] = " "
                    i += 1
            if i < n:
                out[i] = " "
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            if i + 1 < n:
                out[i] = out[i + 1] = " "
                i += 2
            continue
        i += 1
    return "".join(out)


clean = strip_js(tpl)

print("=" * 72)
print("1. every function named by an onclick attribute is defined")
print("=" * 72)
called = sorted(set(re.findall(r'onclick="([a-zA-Z_$][\w$]*)\s*\(', tpl)))
defined = set(re.findall(r"(?:async\s+)?function\s+([a-zA-Z_$][\w$]*)\s*\(", clean))
defined |= set(re.findall(r"(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\(|function)", clean))
print(f"  {len(called)} handlers referenced: {', '.join(called)}")
for fn in called:
    check(f"{fn}() is defined", fn in defined)

print()
print("=" * 72)
print("2. every identifier used in script code is declared or builtin")
print("=" * 72)
BUILTIN = {
    # browser globals
    "document", "window", "navigator", "fetch", "console", "alert", "confirm",
    "prompt", "setTimeout", "clearTimeout", "setInterval", "clearInterval",
    "location", "history", "localStorage", "URL", "URLSearchParams",
    # built-in types and namespaces
    "JSON", "Object", "Array", "Math", "Date", "String", "Number", "Boolean",
    "RegExp", "Error", "Map", "Set", "Promise", "parseInt", "parseFloat",
    "isNaN", "parseFloat", "encodeURIComponent", "Infinity", "NaN",
    # named DOM members accessed via a dot, which the scan below allows through
    "length", "then", "catch", "status", "ok", "text", "json", "headers",
    "body", "value", "style", "classList", "className", "textContent",
    "innerHTML", "innerText", "getElementById", "querySelector",
    "querySelectorAll", "addEventListener", "remove", "forEach", "map",
    "filter", "join", "push", "pop", "slice", "splice", "split", "replace",
    "trim", "toFixed", "toLocaleString", "toLocaleDateString",
    "toLocaleTimeString", "toUpperCase", "toLowerCase", "includes", "indexOf",
    "find", "some", "every", "reduce", "sort", "reverse", "startsWith",
    "endsWith", "padStart", "repeat", "get_data", "as_text", "preventDefault",
    "scrollIntoView", "matches", "test", "stringify", "parse", "keys", "values",
    "entries", "assign", "from", "now", "floor", "ceil", "round", "max", "min",
    "abs", "log", "warn", "error", "info", "debug", "message", "name", "email",
    "appendChild", "removeChild", "closest", "parentNode", "childNodes",
    "firstChild", "nextSibling", "previousSibling", "dataset", "setAttribute",
    "getAttribute", "removeAttribute", "hasAttribute", "open", "close",
    "send", "abort", "clone", "then", "finally", "call", "apply", "bind",
}
# Language keywords are not variables. Without this the scan reports 'if',
# 'return', 'true' and friends as unresolved.
KEYWORDS = {
    "async", "await", "break", "case", "catch", "class", "const", "continue",
    "debugger", "default", "delete", "do", "else", "export", "extends",
    "false", "finally", "for", "function", "if", "import", "in", "instanceof",
    "let", "new", "null", "of", "return", "static", "super", "switch", "this",
    "throw", "true", "try", "typeof", "undefined", "var", "void", "while",
    "with", "yield",
    # The browser injects `event` into inline onclick handlers, so
    # `const btn = event.target` is valid without a declaration.
    "event",
    # '$' is a literal inside template literals, not an identifier.
    "$",
}
# Anything declared anywhere in the script: functions, variables, params.
declared = set(defined)
declared |= set(re.findall(r"\b(?:const|let|var)\s+([a-zA-Z_$][\w$]*)", clean))
for params in re.findall(r"function\s+[\w$]*\s*\(([^)]*)\)", clean):
    for p in params.split(","):
        p = p.strip().split("=")[0].strip().lstrip(".")
        if re.fullmatch(r"[a-zA-Z_$][\w$]*", p or ""):
            declared.add(p)
# `catch (e)` bindings.
declared |= set(re.findall(r"catch\s*\(\s*([a-zA-Z_$][\w$]*)", clean))
# for..of / destructuring targets.
declared |= set(re.findall(r"for\s*\(\s*(?:const|let|var)\s+([a-zA-Z_$][\w$]*)", clean))

# Scan only the script bodies, and only bare identifiers (not after a dot,
# not object-literal keys, not declarations).
script_blocks = re.findall(r"<script[^>]*>(.*?)</script>", tpl, re.DOTALL)
unresolved = []
for block in script_blocks:
    # Only the script body, not the surrounding HTML, so markup words like the
    # "ms" unit in <span class="stat-unit">ms</span> are not scanned.
    cblock = strip_js(block)
    for m in re.finditer(r"(?<![\w.$])([a-zA-Z_$][\w$]*)", cblock):
        ident = m.group(1)
        if ident in BUILTIN or ident in KEYWORDS or ident in declared:
            continue
        before = cblock[:m.start()].rstrip()
        after = cblock[m.end():].lstrip()
        # object literal keys, e.g. { foo: 1 }
        if before.endswith("{") or before.endswith(","):
            continue
        # destructuring targets, e.g. const { a, b } = x  /  ({ p }) =>
        if re.search(r"(?:const|let|var)\s*\{[^}]*$", cblock[:m.start()]):
            continue
        if re.match(r"\s*[,}].*=>", after):
            continue
        # HTML fragments built as strings were stripped, but element names can
        # survive as bare words in template markup inside script blocks.
        if ident in {"div", "span", "td", "tr", "p", "a", "b", "i", "strong",
                     "code", "h1", "h2", "h3", "pre", "ul", "ol", "li", "img",
                     "br", "small", "section", "button", "table", "thead",
                     "tbody", "option", "label", "select", "input", "canvas",
                     "svg", "path", "nav", "main", "header", "footer", "html",
                     "body", "head", "meta", "title", "style", "script",
                     # unit suffixes emitted next to a value inside a template
                     # literal, e.g. ${call.latency_ms}ms
                     "ms", "s", "kb", "mb", "gb", "mss"}:
            continue
        unresolved.append(ident)

unresolved = sorted(set(unresolved))
if unresolved:
    for ident in unresolved:
        check(f"'{ident}' is declared somewhere", False)
else:
    check("no unresolved identifiers in script code", True)
print(f"  scanned {len(script_blocks)} script block(s); "
      f"{len(declared)} declared names")

print()
print("=" * 72)
print("3. the reported regression")
print("=" * 72)
fn = re.search(r"async function showAccountPanel\(\)\s*\{(.*?)\n\}", clean, re.DOTALL)
body = fn.group(1) if fn else ""
decl_at = body.find("const panel")
use_at = body.find("panel.style")
check("showAccountPanel declares its own panel reference", decl_at >= 0)
check("declaration comes before first use", decl_at >= 0 and decl_at < use_at,
      f"decl@{decl_at} use@{use_at}")

print()
print("=" * 72)
print("4. dynamically created element ids")
print("=" * 72)
static_ids = set(re.findall(r'id="([\w-]+)"', tpl))
created = set()
for m in re.finditer(r"\.id\s*=\s*'([\w-]+)'", strip_js(tpl)):
    created.add(m.group(1))
for m in re.finditer(r'id\s*=\s*\"([\w-]+)\"', strip_js(tpl)):
    created.add(m.group(1))
referenced = set(re.findall(r"getElementById\('([\w-]+)'\)", clean))
missing = sorted(referenced - static_ids - created)
check("every getElementById target is either static or created at runtime",
      not missing, str(missing))
for mid in missing:
    print(f"      unresolved id: {mid}")
print(f"  static ids: {len(static_ids)}, runtime-created: {sorted(created)}")

print()
print("=" * 72)
print("5. structural sanity")
print("=" * 72)
check("div tags balanced", len(re.findall(r"<div\b", tpl)) == len(re.findall(r"</div>", tpl)),
      f"{len(re.findall(r'<div', tpl))} vs {len(re.findall(r'</div>', tpl))}")
check("no leftover 'panel' reference before its declaration",
      clean.find("const panel") < clean.find("panel.style")
      if "const panel" in clean else False)

print()
print("FAIL: " + ", ".join(fails) if fails else "PASS: handlers are wired correctly")