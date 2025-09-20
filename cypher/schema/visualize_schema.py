# ontology_visualizer.py
# Render a schema-level (classes + relations) diagram from your ontology
# and optionally an interactive HTML graph (if pyvis is installed).

from pathlib import Path

# ---------- 1) Your ontology (as provided) ----------
node_types = [
    {"label": "CROP", "description": "A cultivated plant grown for food, fiber, or other use (e.g., rice, durian)."},
    {"label": "VARIETY", "description": "A cultivar or variety of a crop."},
    {
        "label": "DISEASE",
        "description": "A disorder affecting a crop or its parts.",
        "properties": [
            {
                "name": "description",
                "type": "STRING",
                "description": "Comprehensive overview of the disease, including cause (biotic/abiotic), key symptoms, severity, progression, and impact on growth and yield; include economic relevance if known.",
            }
        ],
    },
    {
        "label": "CROP_PART",
        "description": "A plant organ (morphological part), e.g., leaf, stem, root, fruit, flower, seed.",
    },
    {
        "label": "SYMPTOM",
        "description": "An observable sign or manifestation of a disease (e.g., leaf spots, cankers, wilting).",
    },
    {
        "label": "PATHOGEN",
        "description": "An organism or agent that causes disease in crops.",
        "properties": [
            {
                "name": "type",
                "type": "STRING",
                "description": "The pathogen type (fungus, bacterium, virus, nematode, abiotic, etc.).",
                "required": True,
            }
        ],
    },
    {
        "label": "CONDITION",
        "description": "An environmental or management condition contributing to disease development (e.g., poor drainage, nutrient imbalance, mechanical injury).",
    },
    {
        "label": "SEASONALITY",
        "description": "Time of year or growth stage when the disease is most likely to occur (e.g., rainy season, flowering).",
    },
    {
        "label": "LOCATION",
        "description": "Geographic area where the disease has been reported (e.g., Southern Vietnam, Eastern Thailand, Sabah, Malaysia).",
    },
    {
        "label": "TREATMENT",
        "description": "An intervention used to manage or cure a disease (e.g., fungicide application, pruning).",
    },
    {
        "label": "PREVENTION_METHOD",
        "description": "A practice that reduces the likelihood of disease (e.g., sanitation, resistant varieties).",
    },
    {
        "label": "SPREAD_METHOD",
        "description": "A transmission pathway by which the disease or pathogen spreads (e.g., wind, insects, irrigation water, tools).",
    },
    {
        "label": "RISK_FACTOR",
        "description": "A factor that increases the likelihood or severity of disease (e.g., high humidity, dense canopy).",
    },
]

relation_types = [
    {"label": "HAS_VARIETY", "description": "Connects a crop to its varieties."},
    {"label": "AFFECTED_BY", "description": "Indicates that a crop is affected by a disease."},
    {"label": "SUSCEPTIBLE_TO", "description": "Indicates that a crop variety is susceptible to a disease."},
    {"label": "HAS_CROP_PART", "description": "Connects a crop to its plant parts."},
    {"label": "HAS_SYMPTOM", "description": "Indicates that a crop part exhibits a symptom."},
    {"label": "HAS_DISEASE", "description": "Indicates that a disease occurs on a specific crop part."},
    {"label": "CAUSED_BY", "description": "Links a disease to its causal agent or contributing condition."},
    {
        "label": "PEAKS_DURING",
        "description": "Links a disease to the season or growth stage when incidence is highest.",
    },
    {"label": "MANAGED_BY", "description": "Links a disease to its management or treatment."},
    {"label": "PREVENTED_BY", "description": "Links a disease to preventive methods."},
    {"label": "SPREADS_VIA", "description": "Links a disease or pathogen to its spread method."},
    {"label": "TRIGGERED_BY", "description": "Links a disease to risk factors that trigger or exacerbate it."},
    {"label": "OCCURS_IN", "description": "Links a disease to locations where it occurs."},
]

allowed_relationships = [
    ("CROP", "HAS_VARIETY", "VARIETY"),
    ("CROP", "AFFECTED_BY", "DISEASE"),
    ("VARIETY", "SUSCEPTIBLE_TO", "DISEASE"),
    ("CROP", "HAS_CROP_PART", "CROP_PART"),
    ("CROP_PART", "HAS_SYMPTOM", "SYMPTOM"),
    ("CROP_PART", "HAS_DISEASE", "DISEASE"),
    ("DISEASE", "CAUSED_BY", "PATHOGEN"),
    ("DISEASE", "CAUSED_BY", "CONDITION"),
    ("DISEASE", "PEAKS_DURING", "SEASONALITY"),
    ("DISEASE", "MANAGED_BY", "TREATMENT"),
    ("DISEASE", "PREVENTED_BY", "PREVENTION_METHOD"),
    ("DISEASE", "SPREADS_VIA", "SPREAD_METHOD"),
    ("DISEASE", "TRIGGERED_BY", "RISK_FACTOR"),
    ("DISEASE", "OCCURS_IN", "LOCATION"),
]

# ---------- 2) Visualization helpers ----------
# Category-based coloring for readability
CATEGORY_COLORS = {
    "BIO": "#C6E9C6",  # crops, varieties, parts, pathogens
    "DISEASE": "#FFD6CC",  # disease, symptom, condition, risk
    "MGMT": "#CCE0FF",  # treatment, prevention
    "CONTEXT": "#E3D7FF",  # location, seasonality, spread
}


def label_category(lbl: str) -> str:
    if lbl in {"CROP", "VARIETY", "CROP_PART", "PATHOGEN"}:
        return "BIO"
    if lbl in {"DISEASE", "SYMPTOM", "CONDITION", "RISK_FACTOR"}:
        return "DISEASE"
    if lbl in {"TREATMENT", "PREVENTION_METHOD"}:
        return "MGMT"
    if lbl in {"LOCATION", "SEASONALITY", "SPREAD_METHOD"}:
        return "CONTEXT"
    return "CONTEXT"


def build_node_lookup(node_types):
    return {n["label"]: n for n in node_types}


# ---------- 3) Graphviz schema rendering ----------
def render_graphviz_schema(node_types, allowed_relationships, out_base="ontology_schema", fmt="png", rankdir="LR"):
    try:
        from graphviz import Digraph
    except ImportError:
        raise SystemExit(
            "Please install graphviz python package and OS-level Graphviz:\n  pip install graphviz\n  (and ensure Graphviz binaries are installed)"
        )

    nodes = build_node_lookup(node_types)
    g = Digraph("OntologySchema", format=fmt)
    g.attr(rankdir=rankdir, splines="spline", concentrate="true", nodesep="0.4", ranksep="0.6")
    g.attr("node", shape="box", style="rounded,filled", color="#555555", fontname="Helvetica")

    # Add nodes with descriptions (first line bold label, then small description)
    for lbl, node in nodes.items():
        cat = label_category(lbl)
        fill = CATEGORY_COLORS.get(cat, "#FFFFFF")
        desc = node.get("description", "")
        # Include properties summary if present
        props = node.get("properties", [])
        props_str = ""
        if props:
            pbits = []
            for p in props:
                required = " (required)" if p.get("required") else ""
                pbits.append(f"• {p['name']}: {p['type']}{required}")
            props_str = "\\n" + "\\n".join(pbits)

        label = f"<<b>{lbl}</b><br/>{desc}{props_str}>"
        g.node(lbl, label=label, fillcolor=fill)

    # Add edges (relation labels)
    for src, rel, dst in allowed_relationships:
        g.edge(src, dst, label=rel, fontsize="10", color="#333333", fontname="Helvetica")

    # Legend cluster
    with g.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", color="#AAAAAA", style="rounded", fontname="Helvetica")
        for name, col in CATEGORY_COLORS.items():
            nid = f"LEG_{name}"
            c.node(nid, label=name, fillcolor=col)
        # tiny invisible edges just for layout
        c.edge("LEG_BIO", "LEG_DISEASE", style="invis")
        c.edge("LEG_MGMT", "LEG_CONTEXT", style="invis")

    out_path = g.render(filename=out_base, cleanup=True)
    print(f"[Graphviz] Wrote {out_path}")
    return out_path


# ---------- 4) Optional: Interactive HTML (pyvis) ----------
def render_interactive_html(node_types, allowed_relationships, out_html="ontology_schema.html"):
    """
    Render an interactive HTML using pyvis.
    - Verifies pyvis and jinja2 are available.
    - Creates parent directories if missing.
    - Uses write_html (no auto-browser) for reliability.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis not installed; skipping interactive HTML. Install with: pip install pyvis")
        return None

    # Ensure Jinja2 is present (pyvis templates depend on it)
    try:
        import jinja2  # noqa: F401
    except ImportError:
        print("jinja2 not installed; install with: pip install jinja2")
        return None

    # Ensure the parent directory exists
    out_path = Path(out_html)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the network
    net = Network(height="800px", width="100%", directed=True, notebook=False, bgcolor="#FFFFFF")
    net.set_options(
        """{
      "physics": {"stabilization": true, "barnesHut": {"gravitationalConstant": -12000}},
      "nodes": {"shape": "box"}
    }"""
    )

    # Category colors (same mapping as before)
    CATEGORY_COLORS = {
        "BIO": "#C6E9C6",
        "DISEASE": "#FFD6CC",
        "MGMT": "#CCE0FF",
        "CONTEXT": "#E3D7FF",
    }

    def label_category(lbl: str) -> str:
        if lbl in {"CROP", "VARIETY", "CROP_PART", "PATHOGEN"}:
            return "BIO"
        if lbl in {"DISEASE", "SYMPTOM", "CONDITION", "RISK_FACTOR"}:
            return "DISEASE"
        if lbl in {"TREATMENT", "PREVENTION_METHOD"}:
            return "MGMT"
        if lbl in {"LOCATION", "SEASONALITY", "SPREAD_METHOD"}:
            return "CONTEXT"
        return "CONTEXT"

    nodes = {n["label"]: n for n in node_types}

    for lbl, node in nodes.items():
        cat = label_category(lbl)
        title = f"<b>{lbl}</b><br>{node.get('description','')}"
        props = node.get("properties", [])
        if props:
            title += (
                "<br><i>Properties</i><ul>"
                + "".join(
                    [f"<li>{p['name']} : {p['type']}{' (required)' if p.get('required') else ''}</li>" for p in props]
                )
                + "</ul>"
            )
        net.add_node(lbl, label=lbl, title=title, color=CATEGORY_COLORS.get(cat, "#FFFFFF"))

    for src, rel, dst in allowed_relationships:
        net.add_edge(src, dst, label=rel, arrows="to")

    # Use write_html for reliability; show() calls write_html then tries to open a browser.
    try:
        net.write_html(str(out_path))
        print(f"[pyvis] Wrote {out_path}")
        return str(out_path)
    except AttributeError as e:
        # Likely due to a missing template/Jinja2
        raise SystemExit(
            "pyvis failed to render HTML (template missing). Ensure Jinja2 is installed:\n"
            "    pip install jinja2\n"
            "If the issue persists, reinstall pyvis: pip install --upgrade --force-reinstall pyvis"
        ) from e


# ---------- 5) CLI ----------
if __name__ == "__main__":
    # Try Graphviz first; if 'dot' is missing, fall back to HTML automatically.
    tried_graphviz = False
    try:
        from graphviz import Digraph  # just to test availability of the python package

        tried_graphviz = True
        # If the OS-level 'dot' is missing, the render call below will raise ExecutableNotFound
        # Reuse your existing render_graphviz_schema(...) function:
        out_png = render_graphviz_schema(
            node_types, allowed_relationships, out_base="docs/ontology_schema", fmt="png", rankdir="LR"
        )
        print(f"[Graphviz] OK: {out_png}")
    except Exception as e:
        print(f"[Graphviz] Skipping PNG/SVG due to: {e}\nFalling back to interactive HTML...")

    out_html = render_interactive_html(node_types, allowed_relationships, out_html="docs/ontology_schema.html")
    if out_html:
        print(f"[HTML] Open in browser: {out_html}")
