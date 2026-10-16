"""
kg.py — the Knowledge Graph layer.

A single networkx.DiGraph holding Course / Concept / Skill / Resource / CurriculumRule
nodes, persisted to JSON (node-link format) so it survives restarts.

Node types (stored as node attribute 'type'):
    course, concept, skill, resource, rule

Edge types (stored as edge attribute 'relation'):
    REQUIRES            course -> concept
    PREREQUISITE_OF     concept -> concept   (source must be learned before target)
    HAS_SKILL           concept -> skill
    HAS_RESOURCE        concept -> resource
    APPLIES_TO          rule -> course
"""
import json
import os
import networkx as nx

KG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db_files", "kg.json")

# ---- Seed curriculum: one course ("Python Basics"), ~12 concepts ----
SEED_COURSE = "python_basics"
SEED_COURSE_NAME = "Python for DBMS Elective"

SEED_CONCEPTS = [
    # (concept_id, display name, difficulty 1-3)
    ("variables", "Variables & Data Types", 1),
    ("operators", "Operators & Expressions", 1),
    ("conditionals", "Conditionals (if/else)", 1),
    ("loops", "Loops (for/while)", 2),
    ("functions", "Functions", 2),
    ("lists", "Lists & Tuples", 2),
    ("dictionaries", "Dictionaries", 2),
    ("strings", "String Handling", 2),
    ("file_io", "File I/O", 2),
    ("exceptions", "Exception Handling", 3),
    ("recursion", "Recursion", 3),
    ("modules", "Modules & Packages", 3),
]

# (source, target) meaning source is a PREREQUISITE_OF target
SEED_PREREQS = [
    ("variables", "operators"),
    ("operators", "conditionals"),
    ("conditionals", "loops"),
    ("variables", "strings"),
    ("loops", "functions"),
    ("variables", "lists"),
    ("lists", "dictionaries"),
    ("functions", "recursion"),
    ("functions", "modules"),
    ("lists", "file_io"),
    ("functions", "exceptions"),
]

SEED_SKILLS = {
    "loops": "Iteration",
    "functions": "Abstraction",
    "recursion": "Recursive thinking",
    "exceptions": "Defensive programming",
}

SEED_RESOURCES = {
    cid: f"{name} — intro notes" for cid, name, _ in SEED_CONCEPTS
}

# CurriculumRule definitions: plain-Python rule objects consumed by verification.py
SEED_RULES = [
    {
        "rule_id": "R1_prereq_order",
        "text": "Prerequisites must precede the concepts that depend on them.",
        "applies_to": SEED_COURSE,
        "rule_kind": "prerequisite_order",
    },
    {
        "rule_id": "R2_max_weekly_load",
        "text": "A plan may not introduce more than 4 new concepts without a checkpoint.",
        "applies_to": SEED_COURSE,
        "rule_kind": "max_load",
        "max_new_concepts": 4,
    },
    {
        "rule_id": "R3_mandatory_concept",
        "text": "Any plan for this course must include 'functions' before completion.",
        "applies_to": SEED_COURSE,
        "rule_kind": "mandatory_concept",
        "concept_id": "functions",
    },
]


def build_seed_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(SEED_COURSE, type="course", name=SEED_COURSE_NAME)

    for cid, name, difficulty in SEED_CONCEPTS:
        g.add_node(cid, type="concept", name=name, difficulty=difficulty)
        g.add_edge(SEED_COURSE, cid, relation="REQUIRES")

    for src, dst in SEED_PREREQS:
        g.add_edge(src, dst, relation="PREREQUISITE_OF")

    for cid, skill_name in SEED_SKILLS.items():
        skill_node = f"skill_{skill_name.lower().replace(' ', '_')}"
        g.add_node(skill_node, type="skill", name=skill_name)
        g.add_edge(cid, skill_node, relation="HAS_SKILL")

    for cid, text in SEED_RESOURCES.items():
        res_node = f"resource_{cid}"
        g.add_node(res_node, type="resource", name=text)
        g.add_edge(cid, res_node, relation="HAS_RESOURCE")

    for rule in SEED_RULES:
        g.add_node(rule["rule_id"], node_type="rule", **rule)
        g.add_edge(rule["rule_id"], rule["applies_to"], relation="APPLIES_TO")

    assert nx.is_directed_acyclic_graph(
        nx.subgraph_view(g, filter_edge=lambda u, v: g.edges[u, v]["relation"] == "PREREQUISITE_OF")
    ), "Prerequisite edges must be acyclic"

    return g


def save_graph(g: nx.DiGraph, path: str = KG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = nx.node_link_data(g, edges="edges")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str = KG_PATH) -> nx.DiGraph:
    if not os.path.exists(path):
        g = build_seed_graph()
        save_graph(g, path)
        return g
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


# ---------- Query helpers used by agents/verification ----------

def get_concepts_for_course(g: nx.DiGraph, course_id: str):
    return [v for u, v, d in g.out_edges(course_id, data=True) if d.get("relation") == "REQUIRES"]


def get_prereq_subgraph(g: nx.DiGraph):
    edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("relation") == "PREREQUISITE_OF"]
    sub = nx.DiGraph()
    sub.add_nodes_from(g.nodes(data=True))
    sub.add_edges_from(edges)
    return sub


def topological_concept_order(g: nx.DiGraph, concept_ids):
    """Return concept_ids ordered so every prerequisite appears before its dependents."""
    prereq_g = get_prereq_subgraph(g)
    sub = prereq_g.subgraph(concept_ids)
    try:
        order = list(nx.topological_sort(sub))
    except nx.NetworkXUnfeasible:
        order = list(concept_ids)
    # topological_sort only orders nodes reachable via edges among concept_ids;
    # make sure every requested concept is present (isolated ones keep original order)
    remaining = [c for c in concept_ids if c not in order]
    return order + remaining


def get_rules_for_course(g: nx.DiGraph, course_id: str):
    rules = []
    for n, d in g.nodes(data=True):
        if d.get("node_type") == "rule" and d.get("applies_to") == course_id:
            rules.append(d)
    return rules


def get_resource_text(g: nx.DiGraph, concept_id: str):
    res_node = f"resource_{concept_id}"
    if res_node in g.nodes:
        return g.nodes[res_node]["name"]
    return f"{concept_id} — study notes"


def concept_name(g: nx.DiGraph, concept_id: str):
    return g.nodes[concept_id].get("name", concept_id) if concept_id in g.nodes else concept_id


def render_png(g: nx.DiGraph, path: str, highlight_known=None):
    """Render a small course-concept subgraph image for the demo/report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    highlight_known = set(highlight_known or [])
    concept_nodes = [n for n, d in g.nodes(data=True) if d.get("type") == "concept"]
    prereq_g = get_prereq_subgraph(g).subgraph(concept_nodes)

    pos = nx.spring_layout(prereq_g, seed=42)
    colors = ["#8BC34A" if n in highlight_known else "#90CAF9" for n in prereq_g.nodes]
    labels = {n: g.nodes[n].get("name", n) for n in prereq_g.nodes}

    plt.figure(figsize=(9, 6))
    nx.draw(prereq_g, pos, with_labels=True, labels=labels, node_color=colors,
            node_size=1800, font_size=7, arrows=True, edge_color="#999999")
    plt.title("Knowledge Graph — Python Basics (green = already known)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


if __name__ == "__main__":
    g = build_seed_graph()
    save_graph(g)
    print(f"Seed KG built: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("Concepts:", get_concepts_for_course(g, SEED_COURSE))
    print("Rules:", [r["rule_id"] for r in get_rules_for_course(g, SEED_COURSE)])
