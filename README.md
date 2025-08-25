## RAG Demo (METU Academic Rules)

A self Retrieval-Augmented Generation (RAG) pipeline that:

- Converts academic rule PDFs in `docs/` into plain text in `docs_text/`
- Chunks content into paragraphs and stores them in a SQLite table
- Generates embeddings with `fastembed` and stores them in `sqlite-vec`
- Retrieves top-k relevant paragraphs for a user query
- Calls a text-generation inference endpoint with an enriched prompt

Required Python 3.11+.

### Folder structure

- `docs/`: Source PDFs
- `docs_text/`: Auto-generated `.txt` files (created by the notebook)
- `rag_training_metu.ipynb`: End-to-end pipeline notebook

### Prerequisites

- Python 3.11 or newer
- pip

### Quickstart

1) Create and activate a virtual environment (Windows cmd):

```bat
python -m venv .venv
.venv\Scripts\activate
```

2) Launch Jupyter and open the notebook:

```bat
pip install -U pip
pip install jupyter
jupyter notebook
```

3) Open `rag_training_metu.ipynb` and run the cells top-to-bottom. The notebook installs exact versions used for this demo:

```text
ipykernel==6.29.5
pandas==2.2.3
pdfminer
pdfminer.six
sqlite-vec==0.1.1
fastembed==0.3.4
```

That’s all you need for a full run using the sample PDFs in `docs/`.

### Pipeline steps (what each notebook section does)

1) Install required libraries
- Installs pinned dependencies listed above to ensure reproducibility.

2) Import libraries and define constants
- Sets database name, table name, and folders:
  - `DB_NAME = "metu_academic"`
  - `TABLE_NAME = "metu_academic_rules"`
  - `ORIG_DOCS_FOLDER = "./docs"`
  - `TEXT_DOCS_FOLDER = "./docs_text"`

3) Convert PDFs to text
- Uses `pdfminer.six` to extract text from every file in `docs/` and saves to `docs_text/<filename>.txt`.

4) Create SQLite database and load `sqlite-vec`
- Creates `metu_academic.db`.
- Loads the `vec0` virtual table extension via `sqlite_vec.load(db)`.

5) Create and populate the metadata table
- Drops and recreates `TABLE_NAME` with schema:

```sql
CREATE TABLE metu_academic_rules (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL
);
```

- Parses each text file into paragraph chunks (filters out very short/empty blocks) and inserts rows `(id, text)`.

6) Generate and store embeddings
- Uses `fastembed.TextEmbedding()` to embed all paragraphs.
- Drops and recreates the vector table with the correct embedding dimension:

```sql
CREATE VIRTUAL TABLE document_embeddings USING vec0(
  id INTEGER PRIMARY KEY,
  embedding FLOAT[DIM]
);
```

- Inserts serialized float32 embeddings using `sqlite_vec.serialize_float32`.

7) Retrieve relevant paragraphs for a query
- Embeds the user query and performs KNN search with `MATCH` and `k = NUM_OF_EXAMPLES`.
- Joins back to paragraph texts and prepares an enriched prompt:
  - Original question
  - “Example information…” section with top-k paragraphs

8) Call the inference endpoint
- Sends the enriched prompt to an HTTP endpoint:
  - URL set via `url = "http://<your_host_URL>:8080/api/inference"` (example)
  - Payload contains `prompt`, `model_id`, and `engine`
- Prints the generated answer.

### Run with your own question

- In the notebook, set:

```python
user_prompt = "<your question>"
NUM_OF_EXAMPLES = 5  # or another k
```

- Re-run from “Search similar documents and generate enriched LLM prompt”.

### Add new PDFs or re-index

1) Drop new PDF files into `docs/`.
2) Re-run from the “Convert PDF documents to .txt documents” cell to regenerate `docs_text/`.
3) Re-run database creation, metadata population, and embedding cells to refresh the index.

### Configuration

- `DB_NAME`: SQLite file name prefix (actual file is `<DB_NAME>.db`).
- `TABLE_NAME`: Metadata table with paragraph texts.
- `ORIG_DOCS_FOLDER`, `TEXT_DOCS_FOLDER`: Input and output folders for document conversion.
- `NUM_OF_EXAMPLES`: Top-k retrieved paragraphs for augmentation.
- `model_id`, `engine`: Model configuration passed to the inference service.

### Troubleshooting

- IProgress warning in Jupyter
  - Install/upgrade widgets: `pip install -U ipywidgets` and refresh the notebook kernel.

- `sqlite-vec` load issues
  - Ensure the version matches `0.1.1` and that the extension is loaded with `sqlite_vec.load(db)` before use.

- Empty or low-quality chunks
  - Adjust paragraph filtering logic in the metadata population cell if your PDFs have unusual formatting.

