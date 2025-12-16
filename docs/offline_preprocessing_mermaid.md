# Offline Preprocessing — Mermaid Diagram

```mermaid
flowchart LR
  A[Database Schema\n(`pre-processing-model/02-schema-database/*.json`)]
  B[Schema Descriptions\n(`pre-processing-model/03-schema-description/*.json`)]

  subgraph TEXT_INDEXING
    C[Text Indexing (SPLADE)\n`04-text_indexing.py`\n`load_schema_items()`]
    D[Embeddings (SPLADE)\n`embed_text()` / `search_schema_splade()`]
  end

  subgraph GRAPH_INDEXING
    E[Knowledge Graph Indexing\n`04-graph-indexing.py`\n(normalisasi: `_norm()`)]
    F[LLM (offline)\n(via `resoning_and_inference/01-query_under_standing.py` wrapper)]
    G[Validation (rule-based)\n(timestamp filter, PK–FK rules, Jaccard desc check)]
  end

  I[Vector Store (Persist)\n`.vector_store/` SPLADE embeddings]
  H[`pre-processing-model/schema_graph.json`\n(Knowledge Graph output)]

  %% flows
  A --> C
  B --> C
  C --> D
  D --> I

  A --> E
  B --> E
  F --> E
  E --> G
  G --> H

  %% hints
  click C "file://D:/Vann/TA (SKRIPSI)/Project/text2sql-cot/pre-processing-model/04-text_indexing.py" "Open text_indexing"
  click E "file://D:/Vann/TA (SKRIPSI)/Project/text2sql-cot/pre-processing-model/04-graph-indexing.py" "Open graph-indexing"

  classDef important fill:#f9f,stroke:#333,stroke-width:1px;
  E,F,G,H class important
```

---

Mapping singkat (node → file / fungsi):
- Database Schema: `pre-processing-model/02-schema-database/*.json` (sumber skema)
- Schema Descriptions: `pre-processing-model/03-schema-description/*.json` (digunakan untuk teks & validasi)
- Text Indexing (SPLADE): `pre-processing-model/04-text_indexing.py` → `load_schema_items()`
- Embeddings (SPLADE): `04-text_indexing.py` → `embed_text()`, `search_schema_splade()`; persist via `build_and_save_vector_store()`
- Knowledge Graph Indexing: `pre-processing-model/04-graph-indexing.py` → `build_schema_graph()` (memuat deskripsi & schema DB)
  - Normalisasi nama: fungsi `_norm()` dalam `04-graph-indexing.py`
  - LLM (offline): `build_schema_graph()` memanggil wrapper LLM dari `resoning_and_inference/01-query_under_standing.py`
  - Validasi rule-based: `_build_rule_based_graph()` dan steps validasi di `build_schema_graph()` (timestamp, PK–FK, Jaccard)
- Output graph: `pre-processing-model/schema_graph.json`

Catatan penting:
- Vektor SPLADE sekarang bisa dipersist (jalankan: `python pre-processing-model/04-text_indexing.py --save-store pre-processing-model/.vector_store`).
- Runtime (query → load vector store → pra-seleksi tabel LLM → ranking kolom → SQL gen) konsumsinya ada di `resoning_and_inference/01-query_under_standing.py`.
- Fallback: jika folder `.vector_store` tidak ada, sistem menghitung embeddings on-the-fly agar kompatibel.

---

Contoh load di runtime:

```python
import importlib.util, sys
spec = importlib.util.spec_from_file_location("text_indexing_module", r"pre-processing-model/04-text_indexing.py")
mod = importlib.util.module_from_spec(spec); sys.modules["text_indexing_module"] = mod; spec.loader.exec_module(mod)
store = mod.load_vector_store(r"pre-processing-model/.vector_store")
```

Untuk regenerasi store:

```powershell
python "pre-processing-model/04-text_indexing.py" --save-store "pre-processing-model/.vector_store"
```