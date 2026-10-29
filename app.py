"""NormalObjects Complaint Desk: a Gradio app on top of normalobjects_langgraph.py.

Run:   python app.py        (then open the local URL it prints, usually http://127.0.0.1:7860)
Needs: OPENAI_API_KEY in .env, and `pip install gradio langgraph langchain-openai python-dotenv`

What you see while a complaint is processed:
- the workflow steps as Christmas lights on the wall, lighting up live as each LangGraph node runs
- a demogorgon running along the loading bar
- a case file with the evidence (lore text + source) behind every answer
- a history of all case files (SQLite)

Fan-made theme for a class lab. Not affiliated with Netflix or any show.
"""
import html
import json
import os

import gradio as gr

import normalobjects_langgraph as nl

# --------------------------------------------------------------------------- step metadata
# label on the wall, caption under the loading bar, bulb colour, progress % at start / end
STEPS = {
    "intake":      ("INTAKE",      "Reading the complaint...",              "#ff3b3b", 6,  20),
    "validate":    ("VALIDATE",    "Checking it against the rules...",      "#ffd23f", 24, 40),
    "investigate": ("INVESTIGATE", "Searching the lore archive...",         "#3bff7a", 44, 66),
    "resolve":     ("RESOLVE",     "Writing the fix...",                    "#3bb2ff", 70, 86),
    "close":       ("CLOSE",       "Closing the case file...",              "#ff5ec8", 90, 100),
    "reject":      ("REJECT",      "Sealing the gate...",                   "#ff1a1a", 90, 100),
}
STANDARD_ORDER = ["intake", "validate", "investigate", "resolve", "close", "reject"]
CATEGORY_COLORS = {"portal": "#3bb2ff", "monster": "#ff8a1e", "psychic": "#ff5ec8", "environmental": "#3bff7a"}

HISTORY_HEADERS = ["#", "Opened", "Category", "Status", "Path", "Evidence", "Complaint"]

EXTRA_EXAMPLE = "When El uses her powers near the portal, the lights flicker and she gets a nosebleed. Why?"


# --------------------------------------------------------------------------- the demogorgon (original SVG)
DEMOGORGON_SVG = """
<svg class="demo" viewBox="0 0 96 72" aria-hidden="true">
  <g class="limb leg back"><path d="M38 44 L30 56 L35 68"/></g>
  <g class="limb arm back"><path d="M56 28 L64 40 L75 43"/></g>
  <path class="torso" d="M36 46 C37 34 47 26 60 23"/>
  <g class="limb leg front"><path d="M38 44 L47 56 L43 68"/></g>
  <g class="limb arm front"><path d="M56 28 L68 33 L80 30"/></g>
  <g class="head">
    <g transform="translate(68 17)">
      <ellipse class="petal" rx="4.6" ry="9" transform="rotate(0) translate(0 -8)"/>
      <ellipse class="petal" rx="4.6" ry="9" transform="rotate(72) translate(0 -8)"/>
      <ellipse class="petal" rx="4.6" ry="9" transform="rotate(144) translate(0 -8)"/>
      <ellipse class="petal" rx="4.6" ry="9" transform="rotate(216) translate(0 -8)"/>
      <ellipse class="petal" rx="4.6" ry="9" transform="rotate(288) translate(0 -8)"/>
      <circle class="mouth" r="4.4"/>
      <circle class="tooth" cx="0" cy="-3.2" r="0.9"/>
      <circle class="tooth" cx="3" cy="-1" r="0.9"/>
      <circle class="tooth" cx="1.9" cy="2.6" r="0.9"/>
      <circle class="tooth" cx="-1.9" cy="2.6" r="0.9"/>
      <circle class="tooth" cx="-3" cy="-1" r="0.9"/>
    </g>
  </g>
</svg>
"""


# --------------------------------------------------------------------------- HTML renderers
def esc(value) -> str:
    return html.escape(str(value or ""))


def render_lights(bulbs: list) -> str:
    """bulbs: list of (label, colour, state) where state is off | active | done | reject | skipped."""
    items = "".join(
        f'<div class="bulb-wrap {state}" style="--c:{color}">'
        f'<div class="socket"></div><div class="bulb"></div>'
        f'<div class="wall-label">{esc(label)}</div></div>'
        for label, color, state in bulbs
    )
    return f"""
<div class="wall">
  <svg class="wire" viewBox="0 0 600 30" preserveAspectRatio="none" aria-hidden="true">
    <path d="M0,6 C50,30 100,30 150,8 C200,30 250,30 300,8 C350,30 400,30 450,8 C500,30 550,30 600,6"/>
  </svg>
  <div class="bulbs">{items}</div>
</div>"""


def standard_bulbs(states: dict) -> list:
    return [(STEPS[k][0], STEPS[k][2], states.get(k, "off")) for k in STANDARD_ORDER]


def render_loader(start: float, end: float, caption: str, mode: str = "running") -> str:
    """mode: idle | running | closed | rejected | error. The bar animates from start% to end%."""
    return f"""
<div class="loader {mode}" style="--from:{start}%;--to:{end}%">
  <div class="track">
    <div class="fill"></div>
    <div class="runner">{DEMOGORGON_SVG}</div>
  </div>
  <div class="caption">{esc(caption)}</div>
</div>"""


def case_field(label: str, value: str, cls: str = "") -> str:
    return f'<div class="field {cls}"><div class="flabel">{esc(label)}</div><div class="fvalue">{value}</div></div>'


def render_case(r: dict, case_id: int) -> str:
    closed = r["status"] == "closed"
    stamp = '<div class="stamp ok">CASE CLOSED</div>' if closed else '<div class="stamp bad">REJECTED</div>'
    fields = [
        case_field("Complaint", f"“{esc(r['complaint'])}”"),
        case_field("Category", esc(r["category"]) or "-"),
        case_field("Workflow path", esc(" → ".join(r["workflow_path"]))),
        case_field("Validation", esc(r["validation_reason"])),
    ]
    if closed:
        fields += [
            case_field("Evidence source", f"<code>{esc(r['evidence_source'])}</code>"),
            case_field("Lore used (verbatim, the ONLY source)", f"<blockquote>{esc(r['evidence'])}</blockquote>", "lore"),
            case_field("Findings (from the lore only)", esc(r["investigation"])),
            case_field("Resolution", f"<strong>{esc(r['resolution'])}</strong>", "resolution"),
        ]
    else:
        fields += [case_field("Outcome", "No investigation, no answer invented. The complaint stopped at the rules.")]
    return f"""
<div class="casefile">
  <div class="tab">HAWKINS · NORMALOBJECTS DIVISION</div>
  <div class="case-head"><span>CASE FILE #{case_id:04d}</span><span>STANDARD WORKFLOW</span></div>
  {stamp}
  {''.join(fields)}
</div>"""


def render_multi_case(m: dict, case_id: int) -> str:
    ok = bool(m["findings"])
    stamp = '<div class="stamp ok">CASE CLOSED</div>' if ok else '<div class="stamp bad">REJECTED</div>'
    findings = "".join(
        f"<li><code>{esc(f.split(']] ', 1)[0] + ']]')}</code> {esc(f.split(']] ', 1)[-1])}</li>" for f in m["findings"]
    ) or "<li>No valid Downside Up category, so nothing was investigated.</li>"
    lore = "".join(
        f"<li><code>LORE['{esc(c)}']</code> “{esc(nl.LORE[c])}”</li>" for c in m["categories"]
    ) or "<li>-</li>"
    return f"""
<div class="casefile">
  <div class="tab">HAWKINS · NORMALOBJECTS DIVISION</div>
  <div class="case-head"><span>CASE FILE #{case_id:04d}</span><span>MULTI-CATEGORY (PARALLEL)</span></div>
  {stamp}
  {case_field("Complaint", f"“{esc(m['complaint'])}”")}
  {case_field("Categories found", esc(", ".join(m["categories"]) or "none"))}
  {case_field("Workflow path", esc(" → ".join(m["workflow_path"])))}
  {case_field("Lore used (the ONLY sources)", f"<ul>{lore}</ul>", "lore")}
  {case_field("Findings, one per category (ran in parallel)", f"<ul>{findings}</ul>", "resolution")}
</div>"""


def render_waiting(text: str) -> str:
    return f"""
<div class="casefile pending">
  <div class="tab">HAWKINS · NORMALOBJECTS DIVISION</div>
  <div class="case-head"><span>CASE FILE #----</span><span>IN PROGRESS</span></div>
  {case_field("Complaint", f"“{esc(text)}”")}
  <div class="typing">Processing<span>.</span><span>.</span><span>.</span></div>
</div>"""


def render_message(title: str, body: str) -> str:
    return f"""
<div class="casefile pending">
  <div class="tab">HAWKINS · NORMALOBJECTS DIVISION</div>
  <div class="case-head"><span>{esc(title)}</span><span></span></div>
  <div class="fvalue">{body}</div>
</div>"""


def load_history() -> list:
    rows = []
    for cid, created, category, status, path, source, complaint in nl.load_complaints(50):
        try:
            path = " → ".join(json.loads(path))
        except (TypeError, ValueError):
            pass
        icon = "✅" if status == "closed" else "❌"
        rows.append([cid, created.replace("T", " "), category, f"{icon} {status}", path, source or "-", complaint])
    return rows


# --------------------------------------------------------------------------- workflow runners (generators)
def run_standard(text: str):
    states, prev, caption = {}, 0, "Opening the case file..."
    yield standard_bulbs(states), render_loader(0, 3, caption), render_waiting(text), gr.skip()
    prev, final = 3, None

    for mode, event in nl.app.stream(nl.new_complaint(text), stream_mode=["tasks", "values"]):
        if mode == "values":
            final = event
            continue
        name = event["name"]
        _, step_caption, _, pct_start, pct_end = STEPS[name]
        if "input" in event:                                  # node starts
            states[name] = "reject" if name == "reject" else "active"
            caption, pct = step_caption, pct_start
        else:                                                 # node finished
            states[name] = "reject" if name == "reject" else "done"
            pct = pct_end
            if name == "validate" and not event["result"]["is_valid"]:
                for skipped in ("investigate", "resolve", "close"):
                    states[skipped] = "skipped"
            if name == "validate" and event["result"]["is_valid"]:
                states["reject"] = "skipped"
        yield standard_bulbs(states), render_loader(prev, pct, caption), gr.skip(), gr.skip()
        prev = pct

    case_id = nl.save_result(final)
    closed = final["status"] == "closed"
    caption = "CASE CLOSED · answer traced to the lore" if closed else "GATE SEALED · complaint rejected by the rules"
    yield (standard_bulbs(states), render_loader(prev, 100, caption, "closed" if closed else "rejected"),
           render_case(final, case_id), load_history())


def run_multi(text: str):
    intake_state, combine_state = "off", "off"
    categories, cat_states, task_cat = [], {}, {}
    prev, caption = 3, "Opening the case file..."

    def bulbs():
        items = [("INTAKE+", STEPS["intake"][2], intake_state)]
        if categories:
            items += [(c.upper(), CATEGORY_COLORS[c], cat_states.get(c, "off")) for c in categories]
        else:
            items += [("INVESTIGATE ×N", STEPS["investigate"][2], "skipped" if intake_state == "done" else "off")]
        items.append(("COMBINE", STEPS["close"][2], combine_state))
        return render_lights(items)

    yield bulbs(), render_loader(0, 3, caption), render_waiting(text), gr.skip()
    final = None

    for mode, event in nl.parallel_app.stream(nl.new_multi_complaint(text), stream_mode=["tasks", "values"]):
        if mode == "values":
            final = event
            continue
        name, starting = event["name"], "input" in event
        if name == "multi_intake":
            intake_state = "active" if starting else "done"
            if not starting:
                categories = event["result"]["categories"]
            caption, pct = "Finding every category that applies...", 6 if starting else 25
        elif name == "investigate_one":
            if starting:
                cat = event["input"]["category"]
                task_cat[event["id"]] = cat
                cat_states[cat] = "active"
                caption = f"Searching {len(task_cat)} lore entries in parallel..."
            else:
                cat_states[task_cat[event["id"]]] = "done"
            done = sum(s == "done" for s in cat_states.values())
            pct = 30 + 55 * done / max(len(categories), 1)
        else:  # combine
            combine_state = "active" if starting else ("done" if categories else "reject")
            caption, pct = "Combining the findings...", 88 if starting else 100
        yield bulbs(), render_loader(prev, pct, caption), gr.skip(), gr.skip()
        prev = pct

    case_id = nl.save_multi_result(final)
    ok = bool(final["findings"])
    caption = ("CASE CLOSED · " + f"{len(final['categories'])} lore entries checked") if ok else \
        "GATE SEALED · no valid category"
    yield bulbs(), render_loader(prev, 100, caption, "closed" if ok else "rejected"), \
        render_multi_case(final, case_id), load_history()


def process(text: str, multi_mode: bool):
    """Gradio handler: streams (lights, loader, case file, history) after every LangGraph event."""
    text = (text or "").strip()
    if not os.getenv("OPENAI_API_KEY"):
        yield (gr.skip(), render_loader(0, 100, "No OPENAI_API_KEY found", "error"),
               render_message("SETUP NEEDED", "Add <code>OPENAI_API_KEY=...</code> to a <code>.env</code> file "
                              "next to app.py and restart."), gr.skip())
        return
    try:
        runner = run_multi if multi_mode else run_standard
        for lights, loader, case, history in runner(text):
            lights = render_lights(lights) if isinstance(lights, list) else lights
            yield lights, loader, case, history
    except Exception as e:  # e.g. network / API errors after retries
        yield (gr.skip(), render_loader(0, 100, "Something in the Downside Up broke the signal", "error"),
               render_message("ERROR", f"<code>{esc(type(e).__name__)}: {esc(e)}</code><br>"
                              "The complaint was not saved. Try again."), gr.skip())


# --------------------------------------------------------------------------- look and feel
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&family=Courier+Prime:wght@400;700&family=Permanent+Marker&family=VT323&display=swap" rel="stylesheet">
"""

CSS = """
:root {
  --st-bg: #070506;
  --st-red: #e8202a;
  --st-red-soft: rgba(232, 32, 42, .55);
  --st-ink: #ece3d4;
  --st-muted: #a39a8e;
  --st-paper: #e9dfc6;
  --st-paper-ink: #2a2118;
}
body, .gradio-container { background: var(--st-bg) !important; color: var(--st-ink); }
.gradio-container { position: relative; overflow-x: hidden; }

/* floating spores + red glow rising from below */
.gradio-container::before {
  content: ""; position: fixed; inset: -50%; pointer-events: none; z-index: 0;
  background-image:
    radial-gradient(1.5px 1.5px at 10% 20%, rgba(255,255,255,.35), transparent 60%),
    radial-gradient(1px 1px at 30% 70%, rgba(255,255,255,.25), transparent 60%),
    radial-gradient(2px 2px at 55% 40%, rgba(255,220,220,.30), transparent 60%),
    radial-gradient(1px 1px at 75% 15%, rgba(255,255,255,.30), transparent 60%),
    radial-gradient(1.5px 1.5px at 90% 80%, rgba(255,255,255,.25), transparent 60%),
    radial-gradient(1px 1px at 45% 90%, rgba(255,255,255,.20), transparent 60%);
  background-size: 420px 420px;
  animation: spores 60s linear infinite;
}
.gradio-container::after {
  content: ""; position: fixed; left: 0; right: 0; bottom: 0; height: 45vh; pointer-events: none; z-index: 0;
  background: radial-gradient(ellipse at 50% 120%, rgba(200, 10, 20, .28), transparent 65%);
}
@keyframes spores { from { transform: translate(0, 0); } to { transform: translate(-420px, -840px); } }
.gradio-container > * { position: relative; z-index: 1; }

/* ---------- title ---------- */
.hero { text-align: center; padding: 28px 8px 6px; }
.hero .line { height: 3px; width: min(520px, 80%); margin: 0 auto; background: var(--st-red);
  box-shadow: 0 0 8px var(--st-red), 0 0 22px var(--st-red-soft); }
.hero h1 {
  font-family: 'Libre Baskerville', Georgia, serif; font-weight: 700;
  font-size: clamp(30px, 7vw, 68px); letter-spacing: .04em; margin: 12px 0 10px; line-height: 1.05;
  color: transparent; -webkit-text-stroke: 1.6px var(--st-red);
  text-shadow: 0 0 10px var(--st-red-soft), 0 0 34px rgba(232, 32, 42, .35);
  animation: neon 6s infinite;
}
.hero .sub { font-family: 'VT323', monospace; font-size: 22px; letter-spacing: .3em; color: var(--st-muted); margin-top: 10px; }
.hero .tagline { font-family: 'Courier Prime', monospace; color: var(--st-ink); opacity: .8; margin-top: 4px; font-size: 14px; }
@keyframes neon {
  0%, 18%, 22%, 62%, 64%, 100% { opacity: 1; }
  20%, 63% { opacity: .55; }
}

/* ---------- alphabet-wall lights ---------- */
.wall { position: relative; padding: 18px 6px 4px; border-radius: 12px;
  background: linear-gradient(180deg, rgba(60, 40, 30, .35), rgba(20, 12, 10, .2));
  border: 1px solid rgba(255, 255, 255, .06); }
.wall .wire { position: absolute; top: 10px; left: 0; width: 100%; height: 30px; }
.wall .wire path { fill: none; stroke: #2c3a2a; stroke-width: 2.5; }
.bulbs { position: relative; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 6px; }
.bulb-wrap { display: flex; flex-direction: column; align-items: center; min-width: 74px; }
.socket { width: 10px; height: 8px; background: #3b3b30; border-radius: 2px; }
.bulb { width: 20px; height: 30px; border-radius: 50% 50% 46% 46% / 60% 60% 40% 40%;
  background: var(--c); opacity: .16; filter: saturate(.5); transition: opacity .3s, box-shadow .3s; }
.wall-label { font-family: 'Permanent Marker', cursive; color: #1b1410; font-size: 15px; margin-top: 10px;
  background: var(--st-paper); padding: 1px 7px; border-radius: 2px; transform: rotate(-2deg); opacity: .55; }
.bulb-wrap:nth-child(even) .wall-label { transform: rotate(2deg); }
.bulb-wrap.done .bulb { opacity: 1; filter: none; box-shadow: 0 0 10px 3px var(--c), 0 0 28px 6px var(--c); }
.bulb-wrap.done .wall-label, .bulb-wrap.active .wall-label, .bulb-wrap.reject .wall-label { opacity: 1; }
.bulb-wrap.active .bulb { filter: none; animation: flicker 1.1s infinite; }
.bulb-wrap.reject .bulb { --c: #ff1a1a; filter: none; animation: alarm .5s infinite alternate; }
.bulb-wrap.skipped .bulb { opacity: .07; filter: grayscale(1); }
.bulb-wrap.skipped .wall-label { opacity: .25; text-decoration: line-through; }
@keyframes flicker {
  0%, 100% { opacity: 1; box-shadow: 0 0 12px 4px var(--c), 0 0 30px 8px var(--c); }
  12% { opacity: .2; box-shadow: none; } 18% { opacity: 1; } 47% { opacity: .35; box-shadow: none; }
  52% { opacity: 1; box-shadow: 0 0 12px 4px var(--c); } 80% { opacity: .6; }
}
@keyframes alarm { from { opacity: .3; box-shadow: none; } to { opacity: 1; box-shadow: 0 0 14px 5px #ff1a1a, 0 0 36px 10px #ff1a1a; } }

/* ---------- loader + demogorgon ---------- */
.loader { padding: 26px 8px 4px; }
.track { position: relative; height: 12px; border-radius: 6px; background: #1a0f11; border: 1px solid #3a1416;
  margin: 70px 44px 0; }
.fill { position: absolute; inset: 0 auto 0 0; border-radius: 6px; width: var(--to);
  background: linear-gradient(90deg, #6d0a10, var(--st-red)); box-shadow: 0 0 10px var(--st-red), 0 0 24px var(--st-red-soft);
  animation: fill-to .9s ease-out; }
.runner { position: absolute; bottom: 4px; left: var(--to); width: 100px; transform: translateX(-60%);
  animation: run-to .9s ease-out; }
@keyframes fill-to { from { width: var(--from); } to { width: var(--to); } }
@keyframes run-to { from { left: var(--from); } to { left: var(--to); } }
.caption { font-family: 'VT323', monospace; font-size: 22px; color: var(--st-ink); text-align: center; margin-top: 10px;
  letter-spacing: .06em; }
.demo { width: 100px; height: 75px; overflow: visible; filter: drop-shadow(0 0 5px rgba(255, 30, 30, .7)); }
.demo .torso { fill: none; stroke: #4a4450; stroke-width: 8; stroke-linecap: round; }
.demo .limb path { fill: none; stroke: #4a4450; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }
.demo .limb.back path { stroke: #2e2a33; }
.demo .petal { fill: #9b2f28; stroke: #5a1511; stroke-width: .8; }
.demo .mouth { fill: #1a0405; }
.demo .tooth { fill: #f3e6cc; }
.demo .limb, .demo .head { transform-box: view-box; }
.demo .leg { transform-origin: 38px 44px; }
.demo .arm { transform-origin: 56px 28px; }
.demo .head { transform-origin: 68px 17px; }
.loader.running .demo { animation: bob .32s ease-in-out infinite alternate; }
.loader.running .leg.front, .loader.running .arm.back { animation: stride .32s ease-in-out infinite alternate; }
.loader.running .leg.back, .loader.running .arm.front { animation: stride .32s ease-in-out infinite alternate-reverse; }
.loader.running .head { animation: chomp .64s ease-in-out infinite; }
@keyframes stride { from { transform: rotate(-28deg); } to { transform: rotate(28deg); } }
@keyframes bob { from { transform: translateY(0); } to { transform: translateY(-4px); } }
@keyframes chomp { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.18) rotate(12deg); } }
.loader.idle .demo { opacity: .5; filter: grayscale(.6); }
.loader.idle .caption { color: var(--st-muted); }
.loader.closed .demo { animation: victory .8s ease-in-out 2; }
.loader.closed .caption { color: #8dffb0; text-shadow: 0 0 10px rgba(80, 255, 150, .5); }
.loader.rejected .fill { background: #ff1a1a; animation: fill-to .9s ease-out, siren .5s 1s infinite alternate; }
.loader.rejected .caption { color: #ff6b6b; text-shadow: 0 0 10px rgba(255, 30, 30, .6); }
.loader.error .caption { color: #ffd23f; }
@keyframes victory { 50% { transform: translateY(-14px) rotate(-8deg); } }
@keyframes siren { from { opacity: .45; } to { opacity: 1; } }

/* ---------- case file ---------- */
.casefile { position: relative; background: var(--st-paper); color: var(--st-paper-ink); border-radius: 4px 10px 10px 10px;
  padding: 22px 22px 18px; margin-top: 26px; font-family: 'Courier Prime', monospace;
  box-shadow: 0 10px 30px rgba(0, 0, 0, .6), inset 0 0 60px rgba(120, 90, 40, .25); }
.casefile .case-head, .casefile .fvalue, .casefile .typing, .casefile li, .casefile strong, .casefile blockquote {
  color: var(--st-paper-ink) !important; }
.casefile .flabel { color: #7a6646 !important; }
.casefile .fvalue code { color: #8a1016 !important; }
.casefile .tab { position: absolute; top: -22px; left: 0; background: #d8c9a3; color: #5a4a30; font-size: 11px;
  letter-spacing: .15em; padding: 4px 14px; border-radius: 6px 6px 0 0; font-weight: 700; }
.case-head { display: flex; justify-content: space-between; border-bottom: 2px dashed rgba(42, 33, 24, .35);
  padding-bottom: 8px; margin-bottom: 12px; font-weight: 700; letter-spacing: .08em; font-size: 14px; gap: 8px; flex-wrap: wrap; }
.field { margin: 10px 0; }
.flabel { font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: #7a6646; font-weight: 700; }
.fvalue { font-size: 15px; line-height: 1.5; overflow-wrap: anywhere; }
.fvalue code { background: rgba(42, 33, 24, .1); padding: 1px 5px; border-radius: 3px; color: #8a1016; }
.fvalue ul { margin: 4px 0 0 18px; padding: 0; }
.field.lore blockquote, .field.lore ul { margin: 6px 0 0; padding: 8px 12px; border-left: 4px solid #b3151c;
  background: rgba(179, 21, 28, .07); }
.field.lore ul { padding-left: 28px; }
.field.resolution .fvalue { background: rgba(255, 255, 255, .35); padding: 8px 12px; border-radius: 4px; }
.stamp { display: inline-block; margin: 4px 0 6px 6px; font-family: 'Libre Baskerville', serif; font-weight: 700;
  font-size: 20px; letter-spacing: .12em; padding: 4px 12px; border: 3px solid; border-radius: 6px;
  transform: rotate(-6deg); opacity: .8; mix-blend-mode: multiply; animation: stamp .35s ease-out; }
.stamp.ok { color: #1f6b35; border-color: #1f6b35; }
.stamp.bad { color: #b3151c; border-color: #b3151c; }
@keyframes stamp { from { transform: rotate(-6deg) scale(2.4); opacity: 0; } to { transform: rotate(-6deg) scale(1); opacity: .8; } }
.casefile.pending { opacity: .9; }
.typing { font-size: 16px; margin-top: 10px; }
.typing span { animation: dot 1.2s infinite; }
.typing span:nth-child(2) { animation-delay: .2s; } .typing span:nth-child(3) { animation-delay: .4s; }
@keyframes dot { 0%, 100% { opacity: .1; } 50% { opacity: 1; } }

/* ---------- inputs ---------- */
/* complaint box header: neon serif title, blinking "recording" light, incident-report tag */
#complaint-box [data-testid="block-info"] {
  display: flex; align-items: center; gap: 12px; width: 100%;
  font-family: 'Libre Baskerville', Georgia, serif !important; font-size: 21px !important;
  font-weight: 700 !important; font-style: italic; letter-spacing: .02em; color: var(--st-ink) !important;
  text-shadow: 0 0 10px var(--st-red-soft), 0 0 2px rgba(255, 255, 255, .3);
  padding: 4px 0 12px; margin-bottom: 12px; border-bottom: 1px solid rgba(232, 32, 42, .4);
  box-shadow: 0 8px 12px -10px var(--st-red);
}
#complaint-box [data-testid="block-info"]::before {
  content: ""; flex: none; width: 11px; height: 11px; border-radius: 50%; background: var(--st-red);
  box-shadow: 0 0 8px var(--st-red), 0 0 18px var(--st-red-soft); animation: rec 1.6s infinite;
}
#complaint-box [data-testid="block-info"]::after {
  content: "INCIDENT REPORT"; margin-left: auto; flex: none; font-family: 'VT323', monospace; font-style: normal;
  font-size: 16px; letter-spacing: .2em; color: var(--st-red); text-shadow: none;
  border: 1px solid var(--st-red-soft); border-radius: 3px; padding: 1px 7px;
}
@keyframes rec { 0%, 60% { opacity: 1; } 61%, 100% { opacity: .2; } }
@media (max-width: 640px) {
  #complaint-box [data-testid="block-info"] { font-size: 17px !important; }
  #complaint-box [data-testid="block-info"]::after { display: none; }
}

#submit-btn { font-family: 'Libre Baskerville', serif; letter-spacing: .12em; font-size: 16px;
  box-shadow: 0 0 14px var(--st-red-soft); }
#submit-btn:hover { box-shadow: 0 0 22px var(--st-red); }
.section-title { font-family: 'VT323', monospace; font-size: 24px; letter-spacing: .2em; color: var(--st-red);
  text-shadow: 0 0 8px var(--st-red-soft); margin: 10px 0 2px; }
.footer-note { text-align: center; color: var(--st-muted); font-size: 12px; font-family: 'Courier Prime', monospace;
  padding: 16px 0 8px; }

@media (max-width: 640px) {
  .bulb-wrap { min-width: 56px; } .wall-label { font-size: 12px; }
  .track { margin: 70px 30px 0; }
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Courier Prime"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#070506", body_background_fill_dark="#070506",
    body_text_color="#ece3d4", body_text_color_dark="#ece3d4",
    body_text_color_subdued="#a39a8e", body_text_color_subdued_dark="#a39a8e",
    block_background_fill="rgba(22, 13, 15, .82)", block_background_fill_dark="rgba(22, 13, 15, .82)",
    block_border_color="#3a1416", block_border_color_dark="#3a1416",
    block_label_text_color="#e8202a", block_label_text_color_dark="#e8202a",
    block_title_text_color="#ece3d4", block_title_text_color_dark="#ece3d4",
    input_background_fill="#120a0c", input_background_fill_dark="#120a0c",
    input_border_color="#4a1a1d", input_border_color_dark="#4a1a1d",
    border_color_primary="#3a1416", border_color_primary_dark="#3a1416",
    background_fill_primary="#0f0809", background_fill_primary_dark="#0f0809",
    background_fill_secondary="#160c0e", background_fill_secondary_dark="#160c0e",
    button_primary_background_fill="#c4151c", button_primary_background_fill_dark="#c4151c",
    button_primary_background_fill_hover="#e3242b", button_primary_background_fill_hover_dark="#e3242b",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#1d1012", button_secondary_background_fill_dark="#1d1012",
    button_secondary_text_color="#ece3d4", button_secondary_text_color_dark="#ece3d4",
    checkbox_background_color_selected="#c4151c", checkbox_background_color_selected_dark="#c4151c",
)

HERO = """
<div class="hero">
  <div class="line"></div>
  <h1>NORMALOBJECTS</h1>
  <div class="line"></div>
  <div class="sub">COMPLAINT DESK · DOWNSIDE UP DIVISION</div>
  <div class="tagline">Every answer follows a fixed path and comes only from the lore. Watch the lights.</div>
</div>
"""

LORE_TABLE = "\n".join(
    ["| Entry | Lore text (the only source the investigator may use) |", "|---|---|"]
    + [f"| `LORE['{k}']` | {v} |" for k, v in nl.LORE.items()]
)


# --------------------------------------------------------------------------- layout
nl.init_db()  # the history table reads from it on page load

with gr.Blocks(title="NormalObjects Complaint Desk") as demo:
    gr.HTML(HERO, container=False, padding=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=300):
            gr.HTML('<div class="section-title">▌ FILE A COMPLAINT</div>', container=False, padding=False)
            complaint_box = gr.Textbox(
                label="What's wrong in the Downside Up?", lines=3, max_lines=6, elem_id="complaint-box",
                placeholder="e.g. Why do the lights flicker every time something is near?")
            multi_mode = gr.Checkbox(
                label="Multi-category mode: investigate every matching category in parallel (Extension E1)",
                value=False)
            submit_btn = gr.Button("OPEN THE CASE", variant="primary", elem_id="submit-btn")
            gr.Examples(examples=[[c] for c in nl.TEST_COMPLAINTS] + [[EXTRA_EXAMPLE]],
                        inputs=[complaint_box], label="Try a known case")
            with gr.Accordion("📼 The lore archive", open=False):
                gr.Markdown(LORE_TABLE)

        with gr.Column(scale=7, min_width=320):
            gr.HTML('<div class="section-title">▌ THE WALL</div>', container=False, padding=False)
            lights = gr.HTML(render_lights(standard_bulbs({})), container=False, padding=False)
            loader = gr.HTML(render_loader(0, 0, "Waiting for a complaint...", "idle"), container=False, padding=False)
            case_file = gr.HTML(render_message(
                "NO OPEN CASE", "File a complaint and its case file appears here, with the exact lore behind the answer."),
                container=False, padding=False)

    gr.HTML('<div class="section-title">▌ CASE FILES</div>', container=False, padding=False)
    history = gr.Dataframe(headers=HISTORY_HEADERS, value=load_history, interactive=False, wrap=True,
                           label="Every processed complaint (saved in complaints.db)")
    refresh_btn = gr.Button("Refresh case files", variant="secondary", size="sm")

    gr.HTML('<div class="footer-note">Built with LangGraph · Fan-made theme for a class lab, '
            'not affiliated with Netflix or any show.</div>', container=False, padding=False)

    outputs = [lights, loader, case_file, history]
    submit_btn.click(process, inputs=[complaint_box, multi_mode], outputs=outputs)
    complaint_box.submit(process, inputs=[complaint_box, multi_mode], outputs=outputs)
    refresh_btn.click(load_history, outputs=history)


if __name__ == "__main__":
    demo.queue().launch(theme=THEME, css=CSS, head=HEAD, server_name="127.0.0.1", inbrowser=True)
