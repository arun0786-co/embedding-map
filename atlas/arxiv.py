"""Reads the arXiv metadata snapshot (Kaggle: Cornell-University/arxiv)
straight from its zip, one JSON object per line."""

import io
import json
import re
import zipfile

# primary category -> one of 15 colour groups. Archives that arXiv later
# merged (alg-geom, chao-dyn, cmp-lg...) go with their successors.
GROUPS = {
    "cs": "computer science", "cmp-lg": "computer science",
    "math": "mathematics", "alg-geom": "mathematics", "dg-ga": "mathematics", "funct-an": "mathematics",
    "q-alg": "mathematics",
    "stat": "statistics", "eess": "electrical engineering", "econ": "economics",
    "q-bio": "quantitative biology", "q-fin": "quantitative finance",
    "astro-ph": "astrophysics", "cond-mat": "condensed matter", "mtrl-th": "condensed matter",
    "supr-con": "condensed matter", "quant-ph": "quantum physics",
    "hep-th": "high energy physics", "hep-ph": "high energy physics", "hep-ex": "high energy physics",
    "hep-lat": "high energy physics", "gr-qc": "general relativity",
    "nucl-th": "nuclear physics", "nucl-ex": "nuclear physics",
    "physics": "other physics", "acc-phys": "other physics", "ao-sci": "other physics", "atom-ph": "other physics",
    "chem-ph": "other physics", "plasm-ph": "other physics", "bayes-an": "other physics",
    "math-ph": "mathematical physics, nonlinear", "nlin": "mathematical physics, nonlinear",
    "chao-dyn": "mathematical physics, nonlinear", "solv-int": "mathematical physics, nonlinear",
    "patt-sol": "mathematical physics, nonlinear", "adap-org": "mathematical physics, nonlinear",
    "comp-gas": "mathematical physics, nonlinear",
}
NAMES = sorted(set(GROUPS.values()))

_ws = re.compile(r"\s+")


def clean(s):
    return _ws.sub(" ", s).strip()


def read(path, limit=None):
    """Yields (id, title, abstract, group index, year, primary category)."""
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".json")][0]
        with z.open(name) as f:
            for k, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
                if limit and k >= limit:
                    return
                r = json.loads(line)
                primary = r["categories"].split()[0]
                group = GROUPS.get(primary.split(".")[0], "other physics")
                year = int(r["versions"][0]["created"].split()[3])
                yield r["id"], clean(r["title"]), clean(r["abstract"]), NAMES.index(group), year, primary
