from pathlib import Path
from disease_schema import node_types, relation_types, allowed_relationships

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
