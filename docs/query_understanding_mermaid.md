# Query Understanding — Mermaid Flowchart

Berikut flowchart Mermaid yang menjelaskan alur "Query Understanding" (blok 3, kotak sebelum Schema Linking) sesuai implementasi di `resoning_and_inference/01-query_under_standing.py`.

```mermaid
flowchart LR
  A[User Query]

  %% Table preselection branch
  A --> B{LLM Table Preselect?}
  B -->|Yes| C[Call LLM preselect\n`get_llm()` via wrapper\n(return table names)]
  B -->|No| D[Use all tables (no preselect)]

  %% Document loading & filtering
  C --> E[Filter documents\n`tables_doc`, `columns_doc`]
  D --> E

  %% Retrieval with SPLADE
  E --> F[SPLADE retrieval\n`embed_text()` → `search_schema_splade()`]
  F --> G[Top‑k candidate columns]

  %% Build metadata and hints
  G --> H[Build column metadata\n`build_column_metadata()`]
  G --> I[Build join hints from graph\n`build_join_hints()`]
  H --> J[Log inputs to `log_print/` (debug/audit)]
  I --> K
  J --> K

  %% Selection prompt to LLM
  K[Construct selection prompt:\n- Candidate list\n- Column descriptions (desc_block)\n- Join hints\n- Selection rules] --> L[Call LLM for column selection\n(streaming via `get_llm()`)]

  %% Extract and fallback
  L --> M[Parse LLM output → `_extract_columns()`]
  M --> N{Valid columns extracted?}
  N -->|Yes| O[Return selected columns → Schema Linking / SQL generation]
  N -->|No| P[Fallback: take top candidates (SPLADE top‑k)]
  P --> O

  %% End
  O --> Q[SQL generation (LLM uses allowed tables/columns + join hints)]

  %% Legend / mapping
  subgraph Metadata[ ]
    direction TB
    X1[Files used:]
    X1 --> |Schema DB| S1[`02-schema-database/*.json` → `load_schema_db()`]
    X1 --> |Schema desc| S2[`03-schema-description/*.json` → `load_schema_items()`]
    X1 --> |Graph| S3[`pre-processing-model/schema_graph.json` → `build_join_hints()`]
  end

  style Metadata fill:#f9f9f9,stroke:#ccc,stroke-dasharray: 3 3

```

**Cara pakai / lokasi file**
- Flowchart ini disimpan di `docs/query_understanding_mermaid.md`.
- Untuk melihatnya, buka file di editor yang mendukung Mermaid atau gunakan renderer Mermaid (online/local).

**Penjelasan singkat tiap node (mengacu fungsi kode)**
- `get_llm()` — inisialisasi dan pemanggilan model LLM via `llama_cpp` dalam `01-query_under_standing.py`.
- `load_schema_items()` — buat dokumen tabel/kolom untuk SPLADE (di `04-text_indexing.py`).
- `search_schema_splade()` / `embed_text()` — lakukan retrieval SPLADE.
- `build_column_metadata()` — muat tipe/deskripsi/contoh tiap kandidat kolom dari deskripsi schema.
- `build_join_hints()` — gunakan `schema_graph.json` (hasil graph indexing) untuk menghasilkan hint join.
- `_extract_columns()` — parse output LLM untuk menghasilkan daftar `table.column` final.

Jika Anda mau, saya bisa:
- Export flowchart ini ke PNG/SVG dan simpan di `docs/`.
- Tambahkan varian vertikal atau lebih terperinci (mis. menampilkan format prompt contoh dan blok logging JSON).

Apa yang mau saya lakukan selanjutnya? (export PNG / lebih detail / tidak perlu)