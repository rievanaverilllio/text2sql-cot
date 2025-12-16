"""
Improved and noise-free schema graph indexing.

Key improvements:
 - Robust timestamp detection (many casings/formats)
 - Forbid id↔id (PK–PK) unless explicit FK pattern exists
 - Validate LLM-proposed semantic links with column descriptions (Jaccard overlap)
 - Fallback to rule-based (but still cleaned) if LLM unavailable or output invalid
"""

import os
import json
import itertools
import importlib.util
import re

BASE = os.path.dirname(__file__)
SCHEMA_DB_DIR = os.path.join(BASE, '02-schema-database')
GRAPH_JSON = os.path.join(BASE, 'schema_graph.json')
LLM_WRAPPER_PATH = os.path.normpath(os.path.join(BASE, '..', 'resoning_and_inference', '01-query_under_standing.py'))
DESC_DIR = os.path.join(BASE, '03-schema-description')

# expanded timestamp patterns (normalize before compare)
TIMESTAMP_PATTERNS = re.compile(r"(created|updated|deleted|timestamp|time|date|modif)", re.IGNORECASE)

# generic column blacklist for naive name-only linking (reject unless desc confirms)
GENERIC_COL_BLACKLIST = {
    "name", "title", "description", "status", "state", "type", "value", "code", "note"
}

def load_schema_db(folder_path: str):
    schema = {}
    if not os.path.isdir(folder_path):
        return schema

    for file in os.listdir(folder_path):
        if not file.endswith('.json'):
            continue

        path = os.path.join(folder_path, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        table_name = data.get('table_name') or data.get('table')
        cols = data.get('columns', [])

        normalized = []
        for c in cols:
            if isinstance(c, dict):
                name = c.get('name')
                normalized.append(name)
            else:
                normalized.append(c)

        if table_name:
            schema[table_name] = [c for c in normalized if c]

    return schema


def _norm(x: str):
    if not x:
        return ""
    return re.sub(r"\s+|-", "_", x.strip().lower())


def _is_timestamp_col(colname: str):
    # normalize: replace camelCase boundaries with underscore then test pattern
    n = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', colname or '')
    return bool(TIMESTAMP_PATTERNS.search(n))


def _jaccard_tokens(a: str, b: str):
    if not a or not b:
        return 0.0
    toks_a = set(re.findall(r"\w+", a.lower()))
    toks_b = set(re.findall(r"\w+", b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    inter = toks_a.intersection(toks_b)
    union = toks_a.union(toks_b)
    return len(inter) / len(union)


def _load_descriptions(desc_dir: str):
    """
    Load schema descriptions (table desc + per-column desc).
    Return: { table: { 'table_description': str, 'columns': {colname: description} } }
    """
    out = {}
    if not os.path.isdir(desc_dir):
        return out
    for fn in os.listdir(desc_dir):
        if not fn.endswith('.json'):
            continue
        p = os.path.join(desc_dir, fn)
        try:
            j = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        tname = j.get('table_name') or j.get('table')
        if not tname:
            continue
        cols = j.get('columns', [])
        col_map = {}
        for c in cols:
            if isinstance(c, dict):
                col_map[c.get('name')] = (c.get('description') or '').strip()
            else:
                col_map[c] = ''
        out[tname] = {
            'table_description': (j.get('description') or '').strip(),
            'columns': col_map
        }
    return out


def _valid_node_format(s: str):
    # must be table.column (both non-empty)
    if not isinstance(s, str):
        return False
    parts = s.split('.', 1)
    return len(parts) == 2 and parts[0].strip() and parts[1].strip()


def build_schema_graph(schema_db: dict):
    """
    Build schema graph using LLM over schema descriptions but validate links:
    - remove timestamp columns
    - forbid id↔id unless explicit FK naming pattern (table_id)
    - accept same-name links only if descriptions confirm similarity (Jaccard >= 0.25)
    Fallback to rule-based if LLM not available / fails.
    """
    # load descriptions for validation
    descs = _load_descriptions(DESC_DIR)

    # try to load LLM wrapper
    try:
        spec = importlib.util.spec_from_file_location("llm_wrapper", LLM_WRAPPER_PATH)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load LLM wrapper module")
        llm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llm_mod)
        llm = llm_mod.get_llm()
    except Exception as e:
        # LLM not available -> fallback to validated rule-based graph
        return _build_rule_based_graph(schema_db, descs)

    # build prompt (concise) containing table, columns and descriptions (if available)
    prompt_lines = [
        "You are a strict schema link detector. Output a single JSON object mapping \"table.column\" -> [\"other.table.col\", ...].",
        "Only output links that indicate a valid join relationship (PK-FK) or clear semantic equivalence confirmed by column descriptions.",
        "Do NOT link timestamps or generic id↔id primary keys unless there is clear FK name (e.g., employee.id ↔ salary.employee_id).",
        "",
        "SCHEMA:"
    ]
    for t, info in descs.items():
        td = info.get('table_description') or ''
        prompt_lines.append(f"TABLE: {t} ; DESC: {td}")
        for col, cdesc in info.get('columns', {}).items():
            prompt_lines.append(f"  - {col}: {cdesc}")
        prompt_lines.append("")

    prompt = "\n".join(prompt_lines)

    # call LLM (try streaming first)
    out = ""
    try:
        for token in llm(prompt, max_tokens=4096, stream=True, temperature=0.0, top_p=0.9, repeat_penalty=1.05):
            out += token.get('choices', [])[0].get('text', '')
    except Exception:
        try:
            resp = llm(prompt, max_tokens=2048, stream=False, temperature=0.0)
            if isinstance(resp, dict):
                out = resp.get('choices', [])[0].get('text', '')
            else:
                out = str(resp)
        except Exception:
            return _build_rule_based_graph(schema_db, descs)

    # extract JSON substring
    out_clean = out.strip()
    first = out_clean.find('{')
    last = out_clean.rfind('}')
    json_text = out_clean[first:last+1] if first != -1 and last != -1 and last > first else out_clean

    try:
        parsed = json.loads(json_text)
    except Exception:
        return _build_rule_based_graph(schema_db, descs)

    # validation rules for parsed links
    cleaned = {}
    for src, nbrs in parsed.items():
        if not _valid_node_format(src):
            continue
        src_tbl, src_col = src.split('.', 1)
        # skip timestamp columns
        if _is_timestamp_col(src_col):
            continue
        if not isinstance(nbrs, list):
            continue
        valid_neighbors = []
        for n in nbrs:
            if not _valid_node_format(n):
                continue
            dst_tbl, dst_col = n.split('.', 1)
            if _is_timestamp_col(dst_col):
                continue
            # avoid same-table
            if dst_tbl == src_tbl:
                continue
            # rule: forbid id↔id (PK-PK) links
            if src_col.lower() == 'id' and dst_col.lower() == 'id':
                # only allow if dst column actually matches pattern "<src_tbl>_id" (rare), else skip
                expected_fk = f"{src_tbl}_id"
                if _norm(dst_col) != _norm(expected_fk):
                    continue
            # allow explicit PK-FK pattern: src 'table.id' <-> other 'table_id'
            if src_col.lower() == 'id' and _norm(dst_col) == f"{_norm(src_tbl)}_id":
                valid_neighbors.append(n)
                continue
            if dst_col.lower() == 'id' and _norm(src_col) == f"{_norm(dst_tbl)}_id":
                valid_neighbors.append(n)
                continue
            # if columns have identical names:
            if _norm(src_col) == _norm(dst_col):
                # require description validation (if both available)
                src_desc = descs.get(src_tbl, {}).get('columns', {}).get(src_col, '') or ''
                dst_desc = descs.get(dst_tbl, {}).get('columns', {}).get(dst_col, '') or ''
                if src_desc and dst_desc:
                    sim = _jaccard_tokens(src_desc, dst_desc)
                    if sim >= 0.25:
                        valid_neighbors.append(n)
                    else:
                        # descriptions disagree -> skip
                        continue
                else:
                    # no descriptions -> only allow if column name is NOT in generic blacklist
                    if _norm(src_col) in GENERIC_COL_BLACKLIST:
                        continue
                    # otherwise (non-generic identical name) accept cautiously
                    valid_neighbors.append(n)
                continue
            # else: if neither PK-FK nor identical names, reject (LLM hallucination)
            continue

        if valid_neighbors:
            cleaned[src] = sorted(list(set(valid_neighbors)))

    # If LLM produced nothing acceptable, fallback to rule-based with same validation
    if not cleaned:
        return _build_rule_based_graph(schema_db, descs)

    return cleaned


def _build_rule_based_graph(schema_db: dict, descs: dict):
    """
    Rule-based graph builder (validated):
     - PK-FK: table.id <-> other.table_id
     - Same-name columns only if descriptions agree OR not generic
     - Remove timestamp columns
    """
    graph = {}
    col_map = {}
    for table, cols in (schema_db or {}).items():
        for col in (cols or []):
            key = _norm(col)
            col_map.setdefault(key, []).append((table, col))

    # Connect same semantic column names (ignore timestamps & validate with descs)
    for key, items in col_map.items():
        # skip timestamp-like names
        if TIMESTAMP_PATTERNS.search(key):
            continue
        if len(items) < 2:
            continue
        for (t1, c1), (t2, c2) in itertools.combinations(items, 2):
            if t1 == t2:
                continue
            # validate identical name: check descriptions if present
            c1_desc = descs.get(t1, {}).get('columns', {}).get(c1, '') or ''
            c2_desc = descs.get(t2, {}).get('columns', {}).get(c2, '') or ''
            if c1_desc and c2_desc:
                if _jaccard_tokens(c1_desc, c2_desc) < 0.25:
                    continue
            else:
                # if both descriptions missing and name is generic -> skip
                if key in GENERIC_COL_BLACKLIST:
                    continue
            n1 = f"{t1}.{c1}"
            n2 = f"{t2}.{c2}"
            graph.setdefault(n1, []).append(n2)
            graph.setdefault(n2, []).append(n1)

    # Smart PK–FK linking
    for table, cols in (schema_db or {}).items():
        pk = None
        for col in cols:
            if isinstance(col, str) and col.lower() == 'id':
                pk = col
                break
        if not pk:
            continue
        pk_node = f"{table}.{pk}"
        for other_table, other_cols in (schema_db or {}).items():
            if other_table == table:
                continue
            fk_name = f"{table}_id"
            for col in other_cols:
                if _norm(col) == _norm(fk_name):
                    fk_node = f"{other_table}.{col}"
                    graph.setdefault(pk_node, []).append(fk_node)
                    graph.setdefault(fk_node, []).append(pk_node)

    # Deduplicate & filter timestamp nodes
    cleaned = {}
    for k, v in list(graph.items()):
        tbl, col = k.split('.', 1) if '.' in k else ('', k)
        if _is_timestamp_col(col):
            continue
        neighbors = [x for x in sorted(set(v)) if not _is_timestamp_col(x.split('.', 1)[1] if '.' in x else x) and x != k]
        if neighbors:
            cleaned[k] = neighbors
    return cleaned


def save_schema_graph(graph: dict, out_path: str = GRAPH_JSON):
    data = {"graph": graph or {}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


def get_schema_graph():
    schema_db = load_schema_db(SCHEMA_DB_DIR)
    return build_schema_graph(schema_db)


if __name__ == "__main__":
    g = get_schema_graph()
    save_schema_graph(g)
    print(f"[INFO] Graph indexing completed → {GRAPH_JSON} created.")
