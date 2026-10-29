"""NormalObjects - Strict Complaint Processor (LangGraph).

Script version of normalobjects_langgraph_draft.ipynb (same state, lore, nodes and graph).

    START -> intake -> validate --(valid)--> investigate -> resolve -> close -> END
                           |
                           +--(invalid)--> reject -> END

Also contains the notebook extensions:
    E1  parallel multi-category investigation   -> parallel_app
    E2  retry policy on the investigate node    -> RETRY_POLICY, build_complaint_graph(retry_policy=...)
    E3  human approval before resolve           -> build_hitl_app()
    E4  SQLite persistence of finished cases    -> init_db(), save_result(), load_complaints()

Run the tests:        python normalobjects_langgraph.py
Run + extension demo: python normalobjects_langgraph.py --extensions
Use from code:        from normalobjects_langgraph import app, new_complaint

Needs OPENAI_API_KEY in a .env file next to this script.
"""
import json
import operator
import os
import sqlite3
import sys
from datetime import datetime
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

load_dotenv()

# temperature=0 -> same input gives (almost) the same output -> consistent, traceable results
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =============================================================== Step 1: state
class ComplaintState(TypedDict):
    complaint: str              # the customer's complaint (input)
    category: str               # intake: portal / monster / psychic / environmental / other
    workflow_path: List[str]    # every node appends its name -> this IS the trace
    status: str                 # new -> intake -> validated|invalid -> investigated -> resolved -> closed|rejected

    is_valid: bool              # validate: does the complaint pass the rules?
    validation_reason: str      # validate: WHY it passed or failed
    evidence: str               # investigate: the exact lore text that was used
    evidence_source: str        # investigate: which lore entry, e.g. "LORE['portal']"
    investigation: str          # investigate: findings, written ONLY from the evidence
    resolution: str             # resolve: short fix, based ONLY on the investigation
    closure: str                # close / reject: final message to the customer


def new_complaint(text: str) -> ComplaintState:
    """Create a fresh starting state for one complaint (all fields present, nothing decided yet)."""
    return {
        "complaint": text,
        "category": "",
        "workflow_path": [],
        "status": "new",
        "is_valid": False,
        "validation_reason": "",
        "evidence": "",
        "evidence_source": "",
        "investigation": "",
        "resolution": "",
        "closure": "",
    }


# =============================================================== Step 2a: lore (the only source)
VALID_CATEGORIES = ["portal", "monster", "psychic", "environmental"]

LORE = {
    "portal": "Portals open when the barrier between worlds is weak. Their timing follows electrical activity, not the clock.",
    "monster": "Demogorgons act alone when hunting, but follow a hive mind when a larger entity controls them.",
    "psychic": "Psychic power drains the user's energy. Bigger or heavier objects need more energy, and nosebleeds are a warning sign.",
    "environmental": "Creatures from the Downside Up disturb electricity: lights flicker and power lines react when they are near.",
}


# =============================================================== Step 2: nodes
def intake_node(state: ComplaintState) -> ComplaintState:
    """Step 1: Intake - Parse and categorize the complaint"""
    print("\n[INTAKE] Processing complaint...")

    complaint = state["complaint"]

    # Categorize complaint using LLM
    categorization_prompt = f"""Categorize this Downside Up complaint into one of these categories:
- portal: Issues with portal timing, location, or behavior
- monster: Issues with creature behavior (demogorgons, etc.)
- psychic: Issues with psychic abilities or limitations
- environmental: Issues with electricity, weather, or physical environment
- other: Anything else

Complaint: {complaint}

Respond with ONLY the category name (portal, monster, psychic, environmental, or other)."""

    response = llm.invoke([HumanMessage(content=categorization_prompt)])
    category = response.content.strip().lower()

    # Update state
    new_state = {
        **state,
        "category": category,
        "workflow_path": state.get("workflow_path", []) + ["intake"],
        "status": "intake"
    }

    print(f"[INTAKE] Categorized as: {category}")
    return new_state


def validate_node(state: ComplaintState) -> ComplaintState:
    """Step 2: Validate - check the complaint against the business rules (no LLM, pure rules)."""
    print("\n[VALIDATE] Checking complaint against rules...")

    raw_category = state["category"]
    # Guardrail: normalize LLM output like "Portal." or "'monster'\n"
    category = raw_category.strip().strip(".'\"`").strip().lower()

    if not state["complaint"].strip():
        is_valid, reason = False, "Rule 1 failed: the complaint is empty."
    elif category in VALID_CATEGORIES:
        is_valid, reason = True, f"Category '{category}' is in VALID_CATEGORIES and has a LORE entry."
    else:
        is_valid, reason = False, (
            f"Rule 2 failed: category '{category}' is not one of {VALID_CATEGORIES}. "
            "NormalObjects only processes Downside Up complaints."
        )

    new_state = {
        **state,
        "category": category,
        "is_valid": is_valid,
        "validation_reason": reason,
        "workflow_path": state.get("workflow_path", []) + ["validate"],
        "status": "validated" if is_valid else "invalid",
    }

    print(f"[VALIDATE] valid={is_valid} | {reason}")
    return new_state


def investigate_node(state: ComplaintState) -> ComplaintState:
    """Step 3: Investigate - gather information from the lore (the only allowed source)."""
    print("\n[INVESTIGATE] Looking up lore...")

    category = state["category"]
    lore = LORE[category]                    # the retrieval / tool step
    source = f"LORE['{category}']"

    investigation_prompt = f"""You are a strict NormalObjects investigator.
Answer the complaint using ONLY the lore below. Do not add any fact that is not in the lore.
If the lore does not answer the complaint, reply exactly: "The lore does not cover this."

Lore ({category}): {lore}

Complaint: {state["complaint"]}

Findings (max 2 sentences):"""

    response = llm.invoke([HumanMessage(content=investigation_prompt)])
    findings = response.content.strip()

    new_state = {
        **state,
        "evidence": lore,
        "evidence_source": source,
        "investigation": findings,
        "workflow_path": state.get("workflow_path", []) + ["investigate"],
        "status": "investigated",
    }

    print(f"[INVESTIGATE] Evidence source: {source}")
    print(f"[INVESTIGATE] Evidence text  : {lore}")
    print(f"[INVESTIGATE] Findings       : {findings}")
    return new_state


def resolve_node(state: ComplaintState) -> ComplaintState:
    """Step 4: Resolve - write a short fix based ONLY on the investigation findings."""
    print("\n[RESOLVE] Writing resolution...")

    resolution_prompt = f"""Write a short fix (1-2 sentences) for the customer.
Base it ONLY on the investigation findings below. Do not add new facts.
If the findings say the lore does not cover this, say the complaint needs a human expert.

Complaint: {state["complaint"]}
Investigation findings: {state["investigation"]}

Resolution:"""

    response = llm.invoke([HumanMessage(content=resolution_prompt)])
    resolution = response.content.strip()

    new_state = {
        **state,
        "resolution": resolution,
        "workflow_path": state.get("workflow_path", []) + ["resolve"],
        "status": "resolved",
    }

    print(f"[RESOLVE] Resolution: {resolution}")
    return new_state


def close_node(state: ComplaintState) -> ComplaintState:
    """Step 5: Close - confirm completion and attach the evidence to the final message."""
    print("\n[CLOSE] Closing complaint...")

    closure = (
        f"✅ Complaint closed.\n"
        f"Category  : {state['category']}\n"
        f"Resolution: {state['resolution']}\n"
        f"Evidence  : {state['evidence_source']} → \"{state['evidence']}\""
    )

    new_state = {
        **state,
        "closure": closure,
        "workflow_path": state.get("workflow_path", []) + ["close"],
        "status": "closed",
    }

    print("[CLOSE] Status: closed")
    return new_state


def reject_node(state: ComplaintState) -> ComplaintState:
    """Invalid path - reject the complaint and explain why (no investigation, no LLM answer)."""
    print("\n[REJECT] Rejecting complaint...")

    closure = f"❌ Complaint rejected. Reason: {state['validation_reason']}"

    new_state = {
        **state,
        "closure": closure,
        "workflow_path": state.get("workflow_path", []) + ["reject"],
        "status": "rejected",
    }

    print(f"[REJECT] {closure}")
    return new_state


# =============================================================== Step 3: graph
def route_after_validation(state: ComplaintState) -> Literal["investigate", "reject"]:
    """Conditional edge: valid complaints get investigated, invalid ones get rejected."""
    if state["is_valid"]:
        return "investigate"
    return "reject"


def build_complaint_graph(investigate_fn=investigate_node, retry_policy=None,
                          checkpointer=None, interrupt_before=None):
    """Step 3 wiring, with optional retry / checkpointer / human-approval settings (E2, E3)."""
    workflow = StateGraph(ComplaintState)

    workflow.add_node("intake", intake_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("investigate", investigate_fn, retry_policy=retry_policy)
    workflow.add_node("resolve", resolve_node)
    workflow.add_node("close", close_node)
    workflow.add_node("reject", reject_node)

    workflow.set_entry_point("intake")                 # Start here (same as add_edge(START, "intake"))
    workflow.add_edge("intake", "validate")            # Intake always goes to validate
    workflow.add_conditional_edges(                    # validate decides: investigate or reject
        "validate",
        route_after_validation,
        {"investigate": "investigate", "reject": "reject"},
    )
    workflow.add_edge("investigate", "resolve")        # Linear flow: investigate -> resolve -> close
    workflow.add_edge("resolve", "close")
    workflow.add_edge("close", END)                    # Close goes to end
    workflow.add_edge("reject", END)                   # Reject goes to end too

    return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


# E2: retry the investigate node on API timeouts (used by the main app below)
RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=0.2, retry_on=ConnectionError)

app = build_complaint_graph(retry_policy=RETRY_POLICY)


# =============================================================== E1: parallel investigation
class MultiState(TypedDict):
    complaint: str
    categories: List[str]
    findings: Annotated[List[str], operator.add]        # parallel nodes APPEND here
    workflow_path: Annotated[List[str], operator.add]   # ...and here
    combined: str


class InvestigateTask(TypedDict):                       # payload of one parallel task
    complaint: str
    category: str


def multi_intake_node(state: MultiState):
    print("\n[MULTI-INTAKE] Finding ALL matching categories...")
    prompt = f"""List ALL categories that apply to this Downside Up complaint, comma-separated.
Choose from: portal, monster, psychic, environmental. If none apply, answer: other

Complaint: {state["complaint"]}

Categories:"""
    raw = llm.invoke([HumanMessage(content=prompt)]).content
    cats = [c.strip(" .'\"\n").lower() for c in raw.split(",")]
    cats = [c for c in dict.fromkeys(cats) if c in VALID_CATEGORIES]   # keep valid, drop duplicates
    print(f"[MULTI-INTAKE] raw={raw!r} → valid categories={cats}")
    return {"categories": cats, "workflow_path": ["multi_intake"]}


def fan_out(state: MultiState):
    """One Send per category -> those investigate_one tasks run in parallel."""
    if not state["categories"]:
        return "combine"                                # nothing valid -> straight to combine (rejects)
    return [Send("investigate_one", {"complaint": state["complaint"], "category": c})
            for c in state["categories"]]


def investigate_one_node(task: InvestigateTask):
    cat = task["category"]
    print(f"[INVESTIGATE:{cat}] using LORE['{cat}']")
    prompt = f"""Answer the complaint using ONLY this lore. Do not add facts.
If the lore does not answer it, reply exactly: "The lore does not cover this."

Lore ({cat}): {LORE[cat]}
Complaint: {task["complaint"]}

Findings (1 sentence):"""
    text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"findings": [f"[LORE['{cat}']] {text}"], "workflow_path": [f"investigate:{cat}"]}


def combine_node(state: MultiState):
    print("[COMBINE] Merging parallel findings...")
    if not state["findings"]:
        combined = "❌ Rejected: no valid Downside Up category."
    else:
        combined = "\n".join(state["findings"])
    return {"combined": combined, "workflow_path": ["combine"]}


def build_parallel_graph():
    parallel = StateGraph(MultiState)
    parallel.add_node("multi_intake", multi_intake_node)
    parallel.add_node("investigate_one", investigate_one_node)
    parallel.add_node("combine", combine_node)
    parallel.add_edge(START, "multi_intake")
    parallel.add_conditional_edges("multi_intake", fan_out, ["investigate_one", "combine"])
    parallel.add_edge("investigate_one", "combine")
    parallel.add_edge("combine", END)
    return parallel.compile()


parallel_app = build_parallel_graph()


def new_multi_complaint(text: str) -> MultiState:
    return {"complaint": text, "categories": [], "findings": [], "workflow_path": [], "combined": ""}


# =============================================================== E3: human approval before resolve
def build_hitl_app():
    """Graph that pauses before `resolve`. Resume with app.invoke(None, config)."""
    return build_complaint_graph(checkpointer=MemorySaver(), interrupt_before=["resolve"])


# =============================================================== E4: SQLite persistence
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "complaints.db")


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, complaint TEXT, category TEXT, status TEXT,
            workflow_path TEXT, evidence_source TEXT, resolution TEXT, closure TEXT)""")


def save_result(r: dict) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO complaints (created_at, complaint, category, status, workflow_path, evidence_source, resolution, closure) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), r["complaint"], r["category"], r["status"],
             json.dumps(r["workflow_path"]), r.get("evidence_source", ""), r.get("resolution", ""), r.get("closure", "")))
        return cur.lastrowid


def save_multi_result(m: dict) -> int:
    """Store a parallel (E1) run in the same table."""
    return save_result({
        "complaint": m["complaint"],
        "category": ", ".join(m["categories"]) or "other",
        "status": "closed" if m["findings"] else "rejected",
        "workflow_path": m["workflow_path"],
        "evidence_source": ", ".join(f"LORE['{c}']" for c in m["categories"]),
        "resolution": m["combined"],
        "closure": m["combined"],
    })


def load_complaints(limit: int = 20) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT id, created_at, category, status, workflow_path, evidence_source, complaint "
            "FROM complaints ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


# =============================================================== Step 4: test run
TEST_COMPLAINTS = [
    "The Downside Up portal opens at different times each day. How do I predict when?",
    "Demogorgons sometimes work together and sometimes fight. What's their deal?",
    "El can move things with her mind but can't lift heavy rocks. Why?",
    "Why do creatures and power lines react so strangely together?",
    "Why the mother at gilmore girls constantly have ups and downs in her mood?"  # Should be rejected
]


def run_tests() -> list:
    print("\nTesting workflow with sample complaints...\n")
    results = []
    for complaint in TEST_COMPLAINTS:
        print("=" * 80)
        print("COMPLAINT:", complaint)
        result = app.invoke(new_complaint(complaint))
        results.append(result)
        save_result(result)
        print("\n" + result["closure"])
        print()

    # Step 5: path of each complaint
    print("\nWORKFLOW PATHS")
    for r in results:
        print(f"{r['status']:<9} {' → '.join(r['workflow_path'])}")
    n_closed = sum(r["status"] == "closed" for r in results)
    n_rejected = sum(r["status"] == "rejected" for r in results)
    print(f"\nclosed={n_closed}, rejected={n_rejected} (expected 4 closed, 1 rejected)")
    return results


def run_extension_demo() -> None:
    print("\n" + "=" * 80 + "\nE1: parallel investigation")
    multi = parallel_app.invoke(new_multi_complaint(
        "When El uses her powers near the portal, the lights flicker and she gets a nosebleed. Why?"))
    print("Path    :", " → ".join(multi["workflow_path"]))
    print("Combined:\n" + multi["combined"])

    print("\n" + "=" * 80 + "\nE3: human approval before resolve")
    hitl_app = build_hitl_app()
    config = {"configurable": {"thread_id": "cli-hitl-001"}}
    hitl_app.invoke(new_complaint(TEST_COMPLAINTS[2]), config)
    snapshot = hitl_app.get_state(config)
    print("Paused before:", snapshot.next, "| evidence:", snapshot.values["evidence_source"])
    print("Findings:", snapshot.values["investigation"])
    if input("Approve resolution? (y/n): ").strip().lower().startswith("y"):
        final = hitl_app.invoke(None, config)
        print(final["closure"])
    else:
        print("Rejected by human reviewer: complaint sent to manual handling.")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY missing: add it to .env")
    init_db()
    run_tests()
    if "--extensions" in sys.argv:
        run_extension_demo()
