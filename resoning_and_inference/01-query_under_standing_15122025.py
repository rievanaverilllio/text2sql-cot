"""Query understanding and SQL generation pipeline using SPLADE + Llama."""

from __future__ import annotations

import atexit
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from llama_cpp import Llama

# =========================
# 1. Paths & configuration
# =========================
MODEL_PATH = Path(r"D:\Vann\TA (SKRIPSI)\Project\text2sql-cot\gguf-van\merged-llama3-instruct-sql.Q8_0.gguf")
BASE = Path(__file__).resolve().parents[1]
TEXT_INDEXING_PATH = BASE / "pre-processing-model" / "04-text_indexing.py"
SCHEMA_DESC_DIR = BASE / "pre-processing-model" / "03-schema-description"
SCHEMA_GRAPH_PATH = BASE / "pre-processing-model" / "schema_graph.json"
VECTOR_STORE_DIR = BASE / "pre-processing-model" / ".vector_store"

# ====================================
# BLOK 1: FINE-TUNING (MODEL LIFECYCLE)
# Training Dataset → LLM Fine Tuned → Runtime Loader
# ====================================
_llm: Optional[Llama] = None
_llm_ctx = 1024  # default context size (can be overridden by CLI)
_last_preselected_tables: Optional[List[str]] = None
_splade_mod = None
_schema_cache: Optional[Dict[str, Any]] = None
_graph_cache: Optional[Dict[str, Any]] = None


# ===============================================
# BLOK 2: PRE-PROCESSING (SCHEMA & INDEXING ASSETS)
# Database Schema → Schema Description / Graph Indexing → Embedding → Text Indexing
# ===============================================


@dataclass
class VectorStoreArtifacts:
    """Light-weight container for vector store assets."""

    tables_meta: List[Dict[str, Any]]
    columns_meta: List[Dict[str, Any]]
    table_embeddings: np.ndarray
    column_embeddings: np.ndarray


# =========================
# 2. LLM lifecycle helpers
# =========================
def get_llm() -> Llama:
    """Initialize the Llama model lazily and reuse the instance."""

    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=_llm_ctx,
            n_threads=16,
            n_batch=512,
            use_mlock=True,
            n_gpu_layers=0,
            verbose=False,
        )
    return _llm


def close_llm() -> None:
    """Release Llama resources at interpreter shutdown."""

    global _llm
    if _llm is not None:
        try:
            _llm.close()
        except Exception:
            pass
        _llm = None


atexit.register(close_llm)


def _stream_llm_text(
    prompt: str,
    max_tokens: int,
    *,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repeat_penalty: float = 1.05,
    interrupt_message: Optional[str] = None,
) -> str:
    """Utility to stream text generations from the fine-tuned LLM."""

    llm = get_llm()
    output_chunks: List[str] = []
    try:
        for token in llm(
            prompt,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        ):
            output_chunks.append(token["choices"][0]["text"])
    except KeyboardInterrupt:
        if interrupt_message:
            print(interrupt_message)
    return "".join(output_chunks)


# =========================
# 3. SPLADE module utilities
# =========================
def _load_splade_module():
    """Load the SPLADE helper module on demand."""

    global _splade_mod
    if _splade_mod is None:
        spec = importlib.util.spec_from_file_location("text_indexing_module", str(TEXT_INDEXING_PATH))
        if spec is None or spec.loader is None:
            raise ImportError(f"Tidak bisa memuat modul SPLADE dari: {TEXT_INDEXING_PATH}")
        _splade_mod = importlib.util.module_from_spec(spec)
        sys.modules["text_indexing_module"] = _splade_mod
        spec.loader.exec_module(_splade_mod)
    return _splade_mod


def _load_schema_documents(mod, use_desc: bool, include_table_desc: bool):
    return mod.load_schema_items(
        str(SCHEMA_DESC_DIR.resolve()),
        use_descriptions=use_desc,
        include_table_desc_in_column_doc=include_table_desc,
        only_description=True,
    )


def _load_vector_store(mod) -> Optional[VectorStoreArtifacts]:
    if not VECTOR_STORE_DIR.exists():
        return None
    try:
        store = mod.load_vector_store(str(VECTOR_STORE_DIR))
    except Exception:
        return None
    return VectorStoreArtifacts(
        tables_meta=store["tables"]["meta"],
        columns_meta=store["columns"]["meta"],
        table_embeddings=store["tables"]["embs"],
        column_embeddings=store["columns"]["embs"],
    )


def _filter_docs_by_tables(tables_doc, columns_doc, allowed_tables: Sequence[str]):
    allowed = set(allowed_tables)
    filtered_tables = [t for t in tables_doc if t["table"] in allowed]
    filtered_columns = [c for c in columns_doc if c["table"] in allowed]
    return filtered_tables, filtered_columns


def _filter_vector_store(vector_store: Optional[VectorStoreArtifacts], allowed_tables: Sequence[str]):
    if vector_store is None:
        return None
    allowed = set(allowed_tables)
    tbl_idxs = [i for i, meta in enumerate(vector_store.tables_meta) if meta["table"] in allowed]
    col_idxs = [i for i, meta in enumerate(vector_store.columns_meta) if meta["table"] in allowed]
    if not tbl_idxs or not col_idxs:
        return None
    return VectorStoreArtifacts(
        tables_meta=[vector_store.tables_meta[i] for i in tbl_idxs],
        columns_meta=[vector_store.columns_meta[i] for i in col_idxs],
        table_embeddings=vector_store.table_embeddings[tbl_idxs, :],
        column_embeddings=vector_store.column_embeddings[col_idxs, :],
    )


def _tables_block_from_description() -> str:
    table_desc_map = _load_schema_description()
    lines = []
    for table, info in table_desc_map.items():
        desc = (info.get("table_description") or "").strip().replace("\n", " ")
        if len(desc) > 180:
            desc = desc[:177] + "..."
        lines.append(f"{table}: {desc}")
    return "\n".join(lines)

 

# =========================
# 4. Schema helpers
# =========================
def _load_schema_description() -> Dict[str, Any]:
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache
    data: Dict[str, Any] = {}
    if SCHEMA_DESC_DIR.exists():
        for json_file in SCHEMA_DESC_DIR.glob("*.json"):
            try:
                content = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            table_name = content.get("table_name")
            columns = content.get("columns", [])
            col_map = {
                col.get("name"): {
                    "type": col.get("type"),
                    "description": col.get("description"),
                    "example": col.get("example"),
                }
                for col in columns
            }
            data[table_name] = {
                "table_description": content.get("description"),
                "columns": col_map,
            }
    _schema_cache = data
    return data


def build_column_metadata(candidates: List[str]) -> List[Dict[str, Any]]:
    schema = _load_schema_description()
    metadata = []
    for col in candidates:
        if "." not in col:
            continue
        table, column = col.split(".", 1)
        meta = schema.get(table, {}).get("columns", {}).get(column, {})
        if meta:
            metadata.append(
                {
                    "name": col,
                    "type": meta.get("type"),
                    "description": meta.get("description"),
                    "example": meta.get("example"),
                }
            )
        else:
            metadata.append({"name": col})
    return metadata


# =========================
# 5. Join graph helpers
# =========================
def _load_schema_graph() -> Dict[str, Any]:
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache
    if not SCHEMA_GRAPH_PATH.exists():
        _graph_cache = {"graph": {}}
        return _graph_cache
    try:
        _graph_cache = json.loads(SCHEMA_GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception:
        _graph_cache = {"graph": {}}
    return _graph_cache


def build_join_hints(candidates: Iterable[str], max_hints: int = 8) -> List[str]:
    graph = _load_schema_graph().get("graph", {})
    candidate_set = set(candidates)
    hints: List[str] = []
    added = set()
    for src, destinations in graph.items():
        src_table = src.split(".", 1)[0] if "." in src else ""
        for dst in destinations:
            dst_table = dst.split(".", 1)[0] if "." in dst else ""
            if not src_table or not dst_table or src_table == dst_table:
                continue
            if src in candidate_set or dst in candidate_set:
                key = tuple(sorted((src, dst)))
                if key in added:
                    continue
                added.add(key)
                hints.append(f"{src} ↔ {dst}")
                if len(hints) >= max_hints:
                    return hints
    return hints


# =========================
# 6. Candidate prioritization helpers
# =========================
def _priority_tables_from_query(query: str) -> set[str]:
    lowered = query.lower()
    targets = set()
    if "karyawan" in lowered:
        targets.add("employee_data")
    if "pelamar" in lowered:
        targets.add("recruitment_data")
    if "pelatihan" in lowered:
        targets.add("training_and_development")
    if "survey karyawan" in lowered:
        targets.add("employee_engagement")
    return targets


def _prioritize_candidates(candidates: List[str], priority_tables: set[str]) -> List[str]:
    if not priority_tables:
        return candidates
    preferred, others = [], []
    for candidate in candidates:
        table = candidate.split(".", 1)[0] if "." in candidate else ""
        (preferred if table in priority_tables else others).append(candidate)
    return preferred + others


# =========================
# 7. Vector store scoring
# =========================
def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    return float(vec_a @ vec_b) / denom


def _score_tables_with_vector_store(
    query: str,
    mod,
    tables_doc: List[Dict[str, Any]],
    columns_doc: List[Dict[str, Any]],
    vector_store: VectorStoreArtifacts,
    top_k: int,
    top_k_tables: int,
    restrict_to_top_tables: bool,
) -> List[str]:
    query_emb = mod.embed_text([query], mod.tokenizer, mod.model)[0]
    table_index = {meta["table"]: idx for idx, meta in enumerate(vector_store.tables_meta)}

    table_scores = []
    for table in tables_doc:
        idx = table_index.get(table["table"])
        if idx is None:
            continue
        emb = vector_store.table_embeddings[idx]
        score = _cosine_similarity(query_emb, emb)
        table_scores.append({"table": table["table"], "score": score})
    top_tables = sorted(table_scores, key=lambda item: item["score"], reverse=True)[:top_k_tables]
    allowed_tables = {entry["table"] for entry in top_tables}

    column_embeddings = {
        (meta["table"], meta["column"]): vector_store.column_embeddings[idx]
        for idx, meta in enumerate(vector_store.columns_meta)
    }
    column_scores = []
    for column in columns_doc:
        if restrict_to_top_tables and column["table"] not in allowed_tables:
            continue
        embedding = column_embeddings.get((column["table"], column["column"]))
        if embedding is None:
            continue
        score = _cosine_similarity(query_emb, embedding)
        column_scores.append(
            {"table": column["table"], "column": column["column"], "score": score}
        )
    ranked_columns = sorted(column_scores, key=lambda item: item["score"], reverse=True)[:top_k]
    return [f"{entry['table']}.{entry['column']}" for entry in ranked_columns]


# =========================
# 9. SPLADE candidate retrieval
# =========================
def _apply_table_preselection(
    query: str,
    tables_doc: List[Dict[str, Any]],
    columns_doc: List[Dict[str, Any]],
    vector_store: Optional[VectorStoreArtifacts],
):
    table_block = _tables_block_from_description()
    select_prompt = f"""
You are a schema reasoning assistant. Select ONLY the tables needed to answer the user's query.
Return table names separated by commas in one line. No explanation.
If one table is sufficient, choose just that table. Include another table ONLY if absolutely required for a join to satisfy the query semantics.

QUERY:
{query}

TABLE DESCRIPTIONS:
{table_block}

Output format example: employee_data or employee_data, recruitment_data
"""
    selection = _stream_llm_text(select_prompt, max_tokens=96)
    first_line = selection.strip().splitlines()[0] if selection.strip() else ""
    picked_tables = [item.strip() for item in first_line.split(",") if item.strip()]

    existing_names = {table["table"] for table in tables_doc}
    picked_valid = [table for table in picked_tables if table in existing_names]
    if not picked_valid:
        picked_valid = list(existing_names)

    filtered_tables, filtered_columns = _filter_docs_by_tables(tables_doc, columns_doc, picked_valid)
    filtered_vector_store = _filter_vector_store(vector_store, picked_valid)
    return filtered_tables, filtered_columns, filtered_vector_store, picked_valid


def get_splade_candidates(
    query: str,
    top_k: int = 10,
    top_k_tables: int = 5,
    restrict_to_top_tables: bool = False,
    splade_use_desc: bool = True,
    splade_include_table_desc_in_col: bool = True,
    llm_table_preselect: bool = True,
    use_vector_store: bool = True,
) -> List[str]:
    """Return relevant schema columns for a natural language query."""

    global _last_preselected_tables

    mod = _load_splade_module()
    tables_doc, columns_doc = _load_schema_documents(mod, splade_use_desc, splade_include_table_desc_in_col)
    vector_store = _load_vector_store(mod) if use_vector_store else None

    if llm_table_preselect:
        tables_doc, columns_doc, vector_store, picked_tables = _apply_table_preselection(
            query, tables_doc, columns_doc, vector_store
        )
        _last_preselected_tables = picked_tables
    else:
        _last_preselected_tables = None

    if vector_store is not None:
        return _score_tables_with_vector_store(
            query,
            mod,
            tables_doc,
            columns_doc,
            vector_store,
            top_k,
            top_k_tables,
            restrict_to_top_tables,
        )

    result = mod.search_schema(
        query,
        tables_doc,
        columns_doc,
        top_k=top_k,
        top_k_tables=top_k_tables,
        top_k_columns=top_k,
        restrict_to_top_tables=restrict_to_top_tables,
    )
    return [f"{item['table']}.{item['column']}" for item in result["relevant_columns"]]


# ===========================================
# BLOK 3: CHAIN OF THOUGHT (QUERY EXECUTION)
# Query User → LLM Fine Tuned → Query Understanding → Schema Linking
# → Value Retrieval Execution → Answer ↔ Validation
# ===========================================

# -------------------------------
# Query Understanding Utilities
# (Query User → Query Understanding)
# -------------------------------
def _collect_candidate_tables(candidates: Iterable[str]) -> List[str]:
    seen = set()
    ordered_tables: List[str] = []
    for candidate in candidates:
        if "." not in candidate:
            continue
        table = candidate.split(".", 1)[0]
        if table and table not in seen:
            ordered_tables.append(table)
            seen.add(table)
    return ordered_tables


def _format_mapping_block(use_static_rules: bool) -> str:
    if not use_static_rules:
        return ""
    return (
        "ATURAN PEMETAAN TABEL (opsional):\n"
        "- \"karyawan\" → employee_data\n"
        "- \"pelamar\" → recruitment_data\n"
        "- \"pelatihan\" → training_and_development\n"
        "- \"survey karyawan\" → employee_engagement"
    )


def _format_join_hints(join_hints: Iterable[str]) -> str:
    hints = list(join_hints)
    return "\n".join(hints) if hints else "(tidak ada hint)"


def _format_column_descriptions(metadata: Sequence[Dict[str, Any]], include_desc: bool) -> str:
    if not include_desc or not metadata:
        return "(deskripsi tidak disertakan)"
    lines = []
    for meta in metadata:
        parts = [meta["name"]]
        if meta.get("type"):
            parts.append(f"tipe={meta['type']}")
        if meta.get("description"):
            description = meta["description"].strip().replace("\n", " ")
            if len(description) > 140:
                description = description[:137] + "..."
            parts.append(description)
        if meta.get("example"):
            example = str(meta["example"]).strip()
            if len(example) > 50:
                example = example[:47] + "..."
            parts.append(f"contoh={example}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _log_prompt_context(payload: Dict[str, Any]) -> None:
    try:
        logs_dir = BASE / "log_print"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now()
        payload["time"] = timestamp.isoformat(timespec="seconds")
        log_file = logs_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        log_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _extract_columns(output: str, candidates: List[str], limit: int) -> List[str]:
    raw_columns = []
    for line in output.splitlines():
        stripped = line.strip().lstrip("-").strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",") if part.strip()]
        for part in parts:
            if "." in part:
                raw_columns.append(part)

    seen = set()
    filtered: List[str] = []
    for column in raw_columns:
        if column in candidates and column not in seen:
            filtered.append(column)
            seen.add(column)
        if len(filtered) >= limit:
            break
    if not filtered:
        filtered = candidates[:limit]
    return filtered


# ---------------------------------------
# Schema Linking & Execution Utilities
# (Schema Linking → Value Retrieval/Answer)
# ---------------------------------------
def _build_selected_descriptions(columns: Sequence[str], include_desc: bool) -> str:
    if not include_desc or not columns:
        return "(tidak ada)"
    schema = _load_schema_description()
    lines = []
    for column in columns:
        if "." not in column:
            continue
        table, name = column.split(".", 1)
        meta = schema.get(table, {}).get("columns", {}).get(name, {})
        if meta:
            description = (meta.get("description") or "").strip().replace("\n", " ")
            if len(description) > 140:
                description = description[:137] + "..."
            lines.append(f"{column} | tipe={meta.get('type')} | {description}")
        else:
            lines.append(column)
    return "\n".join(lines) if lines else "(tidak ada)"


def _append_sql_to_latest_log(sql_text: str) -> None:
    try:
        logs_dir = BASE / "log_print"
        logs = sorted(logs_dir.glob("*.json"))
        if not logs:
            return
        latest = logs[-1]
        content = json.loads(latest.read_text(encoding="utf-8"))
        content["generated_sql"] = sql_text
        latest.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _tables_from_columns(columns: Sequence[str]) -> List[str]:
    seen = set()
    tables: List[str] = []
    for column in columns:
        if "." not in column:
            continue
        table = column.split(".", 1)[0]
        if table and table not in seen:
            tables.append(table)
            seen.add(table)
    return tables


def _clean_sql(sql_text: str) -> str:
    sql_text = sql_text.strip().strip("`")
    lines = [line.strip() for line in sql_text.splitlines() if line.strip()]
    statement: List[str] = []
    in_statement = False
    for line in lines:
        lowered = line.lower()
        if not in_statement and lowered.startswith("select"):
            in_statement = True
        if in_statement:
            if lowered.startswith("select") and statement:
                break
            if line.startswith("```"):
                break
            statement.append(line)
            if line.endswith(";"):
                break
    cleaned = " ".join(statement).strip() or sql_text
    cleaned = cleaned.strip("`")
    if ";" in cleaned:
        cleaned = cleaned.split(";", 1)[0].strip() + ";"
    lowered_all = cleaned.lower()
    first = lowered_all.find("select")
    if first != -1:
        second = lowered_all.find("select", first + 6)
        if second != -1:
            cleaned = cleaned[:second].rstrip()
            if not cleaned.endswith(";"):
                cleaned += ";"
    return cleaned


def _dedup_boolean_conditions(sql: str) -> str:
    try:
        lowered = sql.lower()
        where_idx = lowered.find(" where ")
        if where_idx == -1:
            return sql
        end_positions = [
            lowered.find(keyword, where_idx + 7)
            for keyword in [" group by ", " order by ", " limit ", ";"]
        ]
        end_positions = [pos for pos in end_positions if pos != -1]
        end_idx = min(end_positions) if end_positions else len(sql)

        head = sql[: where_idx + 7]
        where_body = sql[where_idx + 7 : end_idx]
        tail = sql[end_idx:]

        parts = re.split(r"\s+(OR|AND)\s+", where_body, flags=re.IGNORECASE)
        if not parts:
            return sql

        def normalize(condition: str) -> str:
            return re.sub(r"\s+", " ", condition.strip()).lower()

        seen = set()
        conditions: List[str] = []
        connectors: List[str] = []

        first_condition = parts[0].strip()
        if first_condition:
            conditions.append(first_condition)
            seen.add(normalize(first_condition))

        index = 1
        while index + 1 < len(parts):
            connector = parts[index]
            condition = parts[index + 1].strip()
            normalized = normalize(condition)
            if condition and normalized not in seen:
                connectors.append(connector.upper())
                conditions.append(condition)
                seen.add(normalized)
            index += 2

        if not conditions:
            return sql

        rebuilt = conditions[0]
        for idx, condition in enumerate(conditions[1:]):
            rebuilt += f" {connectors[idx]} {condition}"
        return head + rebuilt + tail
    except Exception:
        return sql


# =========================
# 11. Core ask() pipeline
# =========================
def ask(
    query: str,
    top_k: int = 10,
    limit: int = 5,
    max_tokens: int = 128,
    sql_max_tokens: int = 512,
    include_desc: bool = True,
    use_static_rules: bool = False,
    top_k_tables: int = 5,
    restrict_to_top_tables: bool = False,
    splade_use_desc: bool = True,
    splade_include_table_desc_in_col: bool = True,
    use_vector_store: bool = True,
):
    candidates_raw = get_splade_candidates(
        query,
        top_k=top_k,
        top_k_tables=top_k_tables,
        restrict_to_top_tables=restrict_to_top_tables,
        splade_use_desc=splade_use_desc,
        splade_include_table_desc_in_col=splade_include_table_desc_in_col,
        llm_table_preselect=True,
        use_vector_store=use_vector_store,
    )

    candidates = (
        _prioritize_candidates(candidates_raw, _priority_tables_from_query(query))
        if use_static_rules
        else candidates_raw
    )
    metadata = build_column_metadata(candidates) if include_desc else []
    join_hints = build_join_hints(candidates)

    desc_block = _format_column_descriptions(metadata, include_desc)
    join_hints_str = _format_join_hints(join_hints)
    candidates_str = "\n".join(candidates)
    candidate_tables = _collect_candidate_tables(candidates)
    candidate_tables_str = "\n".join(candidate_tables) if candidate_tables else "(none)"
    mapping_block = _format_mapping_block(use_static_rules)

    _log_prompt_context(
        {
            "query": query,
            "top_k": top_k,
            "limit": limit,
            "include_desc": include_desc,
            "use_static_rules": use_static_rules,
            "top_k_tables": top_k_tables,
            "restrict_to_top_tables": restrict_to_top_tables,
            "splade_use_desc": splade_use_desc,
            "splade_include_table_desc_in_col": splade_include_table_desc_in_col,
            "candidate_tables": candidate_tables,
            "mapping_block": mapping_block,
            "llm_preselected_tables": _last_preselected_tables,
            "candidate_tables_str": candidate_tables_str,
            "candidates_str": candidates_str,
            "desc_block": desc_block,
            "join_hints_str": join_hints_str,
            "candidates": candidates,
            "join_hints": join_hints,
        }
    )

    prompt = f"""
Eliminate the table below that is not relevant to the user's goals and needs.
You are a database schema understanding system.

Your tasks:
Read the QUERY and understand its main goal.
Identify the table(s) most relevant to that goal.
Select up to {limit} columns that are directly needed to answer the QUERY.
You MAY select multiple tables if the QUERY requires a JOIN.
Prioritize columns that capture the key attributes or status of the main entity in the QUERY.
Ignore columns that are unrelated to the QUERY’s main meaning.

Table selection rules:
Pick tables that best match the keywords and context in the QUERY.
Only include additional tables if necessary for JOINs (based on JOIN HINTS).
Avoid selecting columns from unrelated tables, even if they appear in the SPLADE top columns.
Output format (single line, comma-separated):
column1, column2, column3

QUERY:
{query}

=== CANDIDATE COLUMNS ===
{candidates_str}

=== COLUMN DESCRIPTIONS ===
{desc_block}

=== JOIN HINTS ===
{join_hints_str}

Important: Focus on columns that describe key attributes, identifiers, and status relevant to the QUERY’s main topic.
"""

    output_text = _stream_llm_text(
        prompt,
        max_tokens=max_tokens,
        repeat_penalty=1.1,
        interrupt_message="\n[Interrupted] Memotong generasi LLM.",
    )
    columns = _extract_columns(output_text, candidates, limit)

    selected_tables = _tables_from_columns(columns)
    selected_desc_block = _build_selected_descriptions(columns, include_desc)

    sql_prompt = f"""
You are a strict text-to-SQL generator for a schema-limited database.

Your job:
Produce EXACTLY ONE valid SQL statement.

HARD CONSTRAINTS (MUST FOLLOW):
1. Output ONLY the SQL statement (no explanations, no markdown, no comments).
2. Use ONLY tables from this allowed list: {', '.join(selected_tables) if selected_tables else '(none)'}.
3. Use ONLY columns from this allowed list: {', '.join(columns) if columns else '(none)'}.
4. NEVER invent table names, column names, or conditions not present in the allowed lists.
5. NEVER add JOIN unless data from multiple tables is explicitly required in the USER QUERY.
6. When the query only asks to list/show/display fields, generate a simple SELECT with NO WHERE.
7. Add WHERE/ORDER BY/GROUP BY/LIMIT ONLY when the USER QUERY explicitly describes a condition.
8. Prefer the simplest correct SQL that satisfies the USER QUERY.
9. If multiple SQL versions could work, output the simplest one.
10. Use these join hints ONLY if the USER QUERY clearly requires multiple tables:
{join_hints_str}

COLUMN INFORMATION:
{selected_desc_block}

USER QUERY:
{query}

Return ONLY the SQL statement.
"""

    sql_text = _stream_llm_text(
        sql_prompt,
        max_tokens=sql_max_tokens,
        repeat_penalty=1.1,
        interrupt_message="\n[Interrupted] Memotong generasi LLM untuk SQL.",
    )
    sql_final = _dedup_boolean_conditions(_clean_sql(sql_text))
    _append_sql_to_latest_log(sql_final)

    return {
        "raw": output_text.strip(),
        "columns": columns,
        "columns_csv": ", ".join(columns),
        "sql": sql_final,
    }


# =========================
# 12. CLI entry point
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Query Understanding + SQL generation (SPLADE + LLM)"
    )
    parser.add_argument("--query", type=str, help="User natural language query", required=False)
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Jumlah kandidat top-k dari SPLADE (kolom yang dipakai)",
    )
    parser.add_argument(
        "--top_k_tables",
        type=int,
        default=5,
        help="Jumlah top-k tabel untuk pembatasan kolom opsional",
    )
    parser.add_argument(
        "--restrict",
        action="store_true",
        help="Batasi kolom hanya dari top tables (berdasarkan top_k_tables)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Jumlah kolom relevan yang diambil")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maksimal token yang dihasilkan LLM",
    )
    parser.add_argument(
        "--sql_max_tokens",
        type=int,
        default=512,
        help="Maksimal token untuk generasi SQL",
    )
    parser.add_argument(
        "--splade-only",
        action="store_true",
        help="Hanya tampilkan kandidat kolom dari SPLADE lalu keluar",
    )
    parser.add_argument("--json", action="store_true", help="Output final dalam JSON (kolom + SQL)")
    parser.add_argument("--no-desc", action="store_true", help="Jangan sertakan deskripsi kolom ke prompt")
    parser.add_argument(
        "--splade-no-desc",
        action="store_true",
        help="Jangan gunakan deskripsi untuk dokumen SPLADE (schema descriptions)",
    )
    parser.add_argument(
        "--splade-no-table-desc-in-col",
        action="store_true",
        help="Jangan masukkan deskripsi tabel ke dokumen kolom SPLADE",
    )
    parser.add_argument(
        "--static-rules",
        action="store_true",
        help="Aktifkan aturan pemetaan statis kata→tabel (opsional)",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Jangan gunakan vector store (recompute embeddings)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=131072,
        help="Context window (n_ctx) untuk model Llama",
    )
    args = parser.parse_args()

    user_query = args.query or input("Masukkan query: ")

    globals()["_llm_ctx"] = args.ctx

    if args.splade_only:
        columns = get_splade_candidates(
            user_query,
            top_k=args.top_k,
            top_k_tables=args.top_k_tables,
            restrict_to_top_tables=args.restrict,
            splade_use_desc=not args.splade_no_desc,
            splade_include_table_desc_in_col=not args.splade_no_table_desc_in_col,
            use_vector_store=not args.no_store,
        )
        print("\nKandidat kolom (SPLADE):")
        for column in columns:
            print(column)
    else:
        result = ask(
            user_query,
            top_k=args.top_k,
            limit=args.limit,
            max_tokens=args.max_tokens,
            sql_max_tokens=args.sql_max_tokens,
            include_desc=not args.no_desc,
            use_static_rules=args.static_rules,
            top_k_tables=args.top_k_tables,
            restrict_to_top_tables=args.restrict,
            splade_use_desc=not args.splade_no_desc,
            splade_include_table_desc_in_col=not args.splade_no_table_desc_in_col,
            use_vector_store=not args.no_store,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result["columns_csv"])
            print("\n--- SQL ---")
            print(result.get("sql", ""))


