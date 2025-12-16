import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---------------------------
# 1. Konfigurasi path
# ---------------------------
BASE = os.path.dirname(__file__)
SCHEMA_DESC_DIR = os.path.join(BASE, "03-schema-description")

# ---------------------------
# 2. Load schema JSON
# ---------------------------
def load_schema_items(path, use_descriptions: bool = True, include_table_desc_in_column_doc: bool = True, max_desc_len: int = 220, only_description: bool = False):
    """
    Load schema description files and build SPLADE-friendly documents.

    - When use_descriptions=True (default), include concise descriptions in texts.
    - When include_table_desc_in_column_doc=True, each column doc also embeds its table description
      to improve query-table alignment.
    - Descriptions are truncated to max_desc_len characters to keep token budgets reasonable.
    """
    tables, columns = [], []
    for f in os.listdir(path):
        if not f.endswith(".json"):
            continue
        data = json.load(open(os.path.join(path, f), encoding="utf-8"))
        table_name = data.get("table_name") or data.get("table") or ""
        table_desc = str(data.get("description", "")).strip()

        def _truncate(txt: str) -> str:
            txt = (txt or "").replace("\n", " ").strip()
            return txt if not use_descriptions else (txt[:max_desc_len] + ("..." if len(txt) > max_desc_len else ""))

        # Build table document
        if only_description:
            # Gunakan hanya deskripsi tabel (fallback ke nama bila kosong)
            t_text = _truncate(table_desc) if table_desc else table_name
        else:
            if use_descriptions and table_desc:
                t_text = f"{table_name}: {_truncate(table_desc)}"
            else:
                t_text = f"{table_name}"
        tables.append({"table": table_name, "text": t_text})

        # Build column documents
        for c in data.get("columns", []):
            col_name = c.get("name", "")
            col_desc = str(c.get("description", "")).strip()
            col_type = str(c.get("type", "")).strip()
            col_example = c.get("example", None)

            if only_description:
                # Hanya deskripsi kolom (+ optional deskripsi tabel) fallback ke nama kolom bila kosong
                desc_parts = []
                if col_desc:
                    desc_parts.append(_truncate(col_desc))
                if include_table_desc_in_column_doc and table_desc:
                    desc_parts.append(_truncate(table_desc))
                if not desc_parts:  # fallback
                    desc_parts.append(f"{table_name}.{col_name}")
                col_text = ". ".join([p for p in desc_parts if p])
            else:
                parts = [f"{table_name}.{col_name}"]
                if use_descriptions:
                    if col_desc:
                        parts.append(_truncate(col_desc))
                    if col_type:
                        parts.append(f"type={col_type}")
                    if col_example is not None and col_example != "":
                        parts.append(f"example={col_example}")
                    if include_table_desc_in_column_doc and table_desc:
                        parts.append(f"table: {_truncate(table_desc)}")
                col_text = ". ".join([p for p in parts if p])
            columns.append({"table": table_name, "column": col_name, "text": col_text})
    return tables, columns

# ---------------------------
# 3. Inisialisasi SPLADE
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
model.eval()

def embed_text(texts, tokenizer, model):
    embeddings = []
    with torch.no_grad():
        for t in texts:
            encoded = tokenizer(t, return_tensors="pt")
            outputs = model(**encoded)
            logits = outputs.logits.squeeze(0)
            emb = F.relu(logits).sum(dim=0).cpu().numpy()
            embeddings.append(emb)
    return np.stack(embeddings)

def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ---------------------------
# 3b. Vector Store (persist)
# ---------------------------
def build_and_save_vector_store(
    output_dir: str,
    schema_desc_dir: str | None = None,
    use_descriptions: bool = True,
    include_table_desc_in_column_doc: bool = True,
    only_description: bool = False,
    max_desc_len: int = 220,
):
    """
    Bangun dan simpan vector store SPLADE ke disk tanpa merusak API lama.

    Files yang dihasilkan di `output_dir`:
    - tables.npz: array embeddings tabel (shape: [N_table, Vocab])
    - columns.npz: array embeddings kolom (shape: [N_col, Vocab])
    - meta.json: mapping teks dokumen dan index → {tables, columns}
    """
    os.makedirs(output_dir, exist_ok=True)
    src_dir = schema_desc_dir or SCHEMA_DESC_DIR
    tables, columns = load_schema_items(
        src_dir,
        use_descriptions=use_descriptions,
        include_table_desc_in_column_doc=include_table_desc_in_column_doc,
        max_desc_len=max_desc_len,
        only_description=only_description,
    )

    table_texts = [t["text"] for t in tables]
    column_texts = [c["text"] for c in columns]

    # Compute embeddings
    table_embs = embed_text(table_texts, tokenizer, model)
    col_embs = embed_text(column_texts, tokenizer, model)

    # Persist
    np.savez(os.path.join(output_dir, "tables.npz"), embs=table_embs)
    np.savez(os.path.join(output_dir, "columns.npz"), embs=col_embs)
    meta = {
        "tables": [{"table": t["table"], "text": t["text"]} for t in tables],
        "columns": [{"table": c["table"], "column": c["column"], "text": c["text"]} for c in columns],
        "splade_model": "naver/splade-cocondenser-ensembledistil",
        "tokenizer": "naver/splade-cocondenser-ensembledistil",
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "tables_count": len(tables),
        "columns_count": len(columns),
        "output_dir": output_dir,
    }

def load_vector_store(output_dir: str):
    """
    Muat vector store dari `output_dir`.
    Return dict: {
      "tables": {"texts": [...], "meta": [...], "embs": np.ndarray},
      "columns": {"texts": [...], "meta": [...], "embs": np.ndarray}
    }
    """
    meta_path = os.path.join(output_dir, "meta.json")
    tables_npz = os.path.join(output_dir, "tables.npz")
    columns_npz = os.path.join(output_dir, "columns.npz")
    if not (os.path.exists(meta_path) and os.path.exists(tables_npz) and os.path.exists(columns_npz)):
        raise FileNotFoundError("Vector store tidak lengkap. Pastikan meta.json, tables.npz, columns.npz ada.")
    meta = json.load(open(meta_path, encoding="utf-8"))
    t_embs = np.load(tables_npz)["embs"]
    c_embs = np.load(columns_npz)["embs"]

    t_texts = [t["text"] for t in meta.get("tables", [])]
    c_texts = [c["text"] for c in meta.get("columns", [])]

    return {
        "tables": {"texts": t_texts, "meta": meta.get("tables", []), "embs": t_embs},
        "columns": {"texts": c_texts, "meta": meta.get("columns", []), "embs": c_embs},
        "info": {"model": meta.get("splade_model"), "tokenizer": meta.get("tokenizer")}
    }

# ---------------------------
# 4. SPLADE search
# ---------------------------
def search_schema_splade(
    query,
    tables,
    columns,
    top_k_tables: int = 2,
    top_k_columns: int = 10,
    restrict_to_top_tables: bool = False,
):
    table_texts = [t["text"] for t in tables]
    column_texts = [c["text"] for c in columns]

    # Catat: jika vector store tersedia, pengguna bisa memanggil fungsi pencarian
    # dengan embeddings yang dimuat; di sini tetap compute on-the-fly agar backward-compatible.
    table_embs = embed_text(table_texts, tokenizer, model)
    col_embs = embed_text(column_texts, tokenizer, model)
    query_emb = embed_text([query], tokenizer, model)[0]

    table_scores = [{"table": t["table"], "score": cosine_similarity(query_emb, e)}
                    for t, e in zip(tables, table_embs)]
    top_tables = sorted(table_scores, key=lambda x: x["score"], reverse=True)[:top_k_tables]

    # Pilih sumber kolom: dibatasi top tables atau seluruh kolom
    if restrict_to_top_tables:
        top_table_names = {t["table"] for t in top_tables}
        col_iter = [c for c in columns if c["table"] in top_table_names]
        col_emb_iter = [e for c, e in zip(columns, col_embs) if c["table"] in top_table_names]
    else:
        col_iter = columns
        col_emb_iter = col_embs

    col_scores_temp = [
        {"table": c["table"], "column": c["column"], "score": cosine_similarity(query_emb, e)}
        for c, e in zip(col_iter, col_emb_iter)
    ]

    # Hapus duplikat kolom
    seen = set()
    col_scores = []
    for c in col_scores_temp:
        key = (c["table"], c["column"])
        if key not in seen:
            col_scores.append(c)
            seen.add(key)

    top_columns_scored = sorted(col_scores, key=lambda x: x["score"], reverse=True)[:top_k_columns]

    # Hasil akhir TANPA skor dan TANPA output top tables
    top_columns = [{"table": c["table"], "column": c["column"]} for c in top_columns_scored]

    # Optional: tampilkan hanya kolom (tanpa skor)
    print("\n=== SPLADE - Top Columns ===")
    for c in top_columns:
        print(f"- {c['table']}.{c['column']}")

    return top_columns

# ---------------------------
# 5. Pipeline search
# ---------------------------
def search_schema(query, tables, columns, top_k=5, top_k_tables: int = 5, top_k_columns: int | None = None, restrict_to_top_tables: bool = False):
    """
    Catatan kompatibilitas:
    - Parameter lama `top_k` dipetakan ke jumlah kolom yang ingin diambil jika `top_k_columns` tidak diberikan.
    """
    if top_k_columns is None:
        # Jika user memberi top_k kecil, paksa minimal 10 agar hasil kolom lebih banyak secara default
        top_k_columns = max(int(top_k), 10)

    top_columns = search_schema_splade(
        query,
        tables,
        columns,
        top_k_tables=top_k_tables,
        top_k_columns=top_k_columns,
        restrict_to_top_tables=restrict_to_top_tables,
    )
    return {"relevant_columns": top_columns}

# ---------------------------
# 6. Test
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="how many female applicants")
    parser.add_argument("--top_k_tables", type=int, default=2)
    parser.add_argument("--top_k_columns", type=int, default=10)
    parser.add_argument("--restrict", action="store_true", help="Batasi kolom hanya dari top tables")
    parser.add_argument("--no-desc", action="store_true", help="Jangan gunakan deskripsi untuk dokumen SPLADE")
    parser.add_argument("--no-table-desc-in-col", action="store_true", help="Jangan masukkan deskripsi tabel ke dokumen kolom")
    parser.add_argument("--save-store", type=str, default="", help="Jika diisi, simpan vector store ke folder ini dan keluar")
    args = parser.parse_args()

    tables, columns = load_schema_items(
        SCHEMA_DESC_DIR,
        use_descriptions=(not args.no_desc),
        include_table_desc_in_column_doc=(not args.no_table_desc_in_col)
    )

    # Opsi: hanya bangun dan simpan vector store
    if args.save_store:
        info = build_and_save_vector_store(
            output_dir=args.save_store,
            schema_desc_dir=SCHEMA_DESC_DIR,
            use_descriptions=(not args.no_desc),
            include_table_desc_in_column_doc=(not args.no_table_desc_in_col),
        )
        print(json.dumps(info, ensure_ascii=False))
        raise SystemExit(0)

    out = search_schema(
        args.query,
        tables,
        columns,
        top_k=args.top_k_columns,  # kompat: treat as columns target
        top_k_tables=args.top_k_tables,
        top_k_columns=args.top_k_columns,
        restrict_to_top_tables=args.restrict,
    )
