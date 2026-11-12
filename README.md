# SEC Filing Co-pilot

Retrieval-augmented Q&A over 10-K / 10-Q / ESG filings, built so that a
financial analyst can act on the answer: every claim carries a page number and
a verbatim snippet, and the service **refuses to return an answer it cannot
prove against the source text**.

General-purpose RAG fails on filings in two specific ways. This project targets
both.

| Failure mode | What goes wrong | What this does instead |
|---|---|---|
| Scrambled tables | A fixed-size splitter cuts a balance sheet mid-row, severing `2,879` from `Operating income` and from its `FY2023` column. The retriever then serves a chunk in which no figure means anything. | Tables are parsed as tables, never merged with prose, never split mid-row. Oversized tables split on row boundaries with the header repeated. |
| Confident wrong numbers | The model produces a fluent sentence containing a figure that is not in the filing, with a plausible-looking citation attached. | Nothing the model says about its own grounding is trusted. Citations and figures are re-checked mechanically against the retrieved text, and an unprovable answer is withheld. |

## Quick start

```bash
pip install -r requirements.txt

# Parse, chunk, embed and index a filing
python -m sec_copilot.cli ingest sample_data/ \
    --company "Acme Industries, Inc." --ticker ACME --form-type 10-K --fiscal-year 2023

# Retrieval only - no model call, no cost
python -m sec_copilot.cli search "operating income fiscal 2023" -k 3

# Full answer with verified citations (requires ANTHROPIC_API_KEY)
python -m sec_copilot.cli ask "What was operating income in fiscal 2023, and how did it move versus 2022?"

# Read the filing's own XBRL tags - exact figures, no retrieval, no model
python -m sec_copilot.cli facts filings/aapl-10k-2025.htm --concept OperatingIncomeLoss

# Same thing over HTTP
uvicorn sec_copilot.api:app --reload   # then POST /ask, /search, /ingest; GET /health, /usage
```

Against real filings rather than the bundled sample:

```bash
python -m evals.fetch_filings          # Apple FY24/FY25 + Microsoft 10-Ks from EDGAR
python -m sec_copilot.cli ingest filings/aapl-10k-2025.htm     --company "Apple Inc." --ticker AAPL --form-type 10-K --fiscal-year 2025
python -m sec_copilot.cli ask --ticker AAPL "How did operating income move year over year?"
```

No API key? Everything except the generation step runs offline:

```bash
python -m sec_copilot.cli --embedding-backend hashing ask --offline "operating income"
```

## The two things that make it work

### 1. Table-aware chunking

`sec_copilot/chunking/table_aware.py`, fed by loaders that keep layout.

* **`PdfPlumberLoader`** finds each table's bounding box and *removes that
  region from the page's prose stream*, so a statement never leaks into text
  chunks as a flat run of digits. Ruled-table detection first, whitespace
  alignment as fallback.
* **`SecHtmlLoader`** walks EDGAR HTML in document order, drops the spacer and
  bare `$` columns filings pad statements with, and reconstructs page numbers
  from explicit page breaks.
* Tables are serialised to **markdown**, which binds every cell to its column
  label on one line. A model reading `| Operating income | 2,879 | 2,113 |`
  cannot attribute the figure to the wrong fiscal year.
* Each chunk is prefixed with `[Company | Form | Item 7. MD&A]`, which is what
  makes a bare balance sheet retrievable by the name of the statement it
  belongs to.

The invariant is tested directly, in `tests/test_chunking.py`: every source row
survives exactly once, every figure stays ordered within its row, and no chunk
ends mid-row.

### 2. Guardrails that actually block

Three layers, in `sec_copilot/generation/`:

**Structural.** The response is generated with `client.messages.parse` against a
pydantic `GroundedAnswer` schema. An answer without a `claims` array, or a claim
without a citation carrying `chunk_id`, `page` and a verbatim `snippet`, is not
a representable value. Citation is not a request the model can quietly decline.

**Prompt.** The system prompt (`prompts.py`) is a frozen constant - it is the
cache prefix for every request - stating that refusal is a correct outcome, that
snippets must be copied character-for-character, and that a calculated figure
must declare its inputs in `derived_from`.

**Mechanical, and this is the one that counts.** `CitationVerifier` re-checks
the finished answer against the retrieved text:

* each snippet must appear verbatim (after whitespace normalisation) in the
  chunk it names, and that chunk must have actually been retrieved;
* the cited page must match the chunk's page;
* **every figure must appear in the retrieved context**, or be declared as
  calculated with grounded inputs;
* **every figure must belong to the period the claim is about**;
* figures are compared on *value*, not digits, so a correct restatement across
  scales is accepted.

The last two matter more than they sound. Checking only that a figure appears
somewhere in the text cannot catch a **column slip** — in a three-year income
statement 133,050, 123,216 and 114,301 are all "in the context", so an answer
giving FY2024's operating income for a FY2025 question passes every text-level
check while being wrong in the way an analyst can least afford. Figures are
therefore located by *cell* (row label plus dated column header), and
cross-checked against the filer's own inline-XBRL tagging. Conversely,
digit-matching used to reject `$133.05 billion` as ungrounded when the statement
said `133,050` under a "(in millions)" heading — correct work, wrongly blocked.

| Case | Verdict |
|---|---|
| Correct figure, correct year | pass |
| **Real figure, wrong fiscal year** | **blocked** — "belongs to 2024, but the claim is about 2025" |
| Correct figure restated in billions | pass |
| Prior-year figure, correctly labelled | pass |
| Fabricated figure | blocked |
| Calculated margin with declared inputs | pass |

See `tests/test_period_guardrail.py` and
`tests/test_guardrails.py::test_invented_figure_is_caught_even_with_a_valid_citation`.

### Inline XBRL: the filing's own ground truth

EDGAR primary documents are inline XBRL, so every reported figure is already
tagged with its concept, exact period, unit and decimal scale:

```xml
<ix:nonFraction name="us-gaap:OperatingIncomeLoss" contextRef="fy2025"
                unitRef="usd" scale="6">133,050</ix:nonFraction>
```

`ingestion/xbrl.py` reads those (919 facts from Apple's FY2025 10-K, 1,577 from
Microsoft's) and the numeric guardrail uses them as authoritative about a
figure's period — the filer tagged it, rather than us inferring it from a
rendered column. It is also directly queryable, with no retrieval and no model:

```bash
python -m sec_copilot.cli facts filings/aapl-10k-2025.htm --concept OperatingIncomeLoss
#   Operating Income Loss (2024-09-29..2025-09-27) = 1.3305e+11 usd
python -m sec_copilot.cli facts filings/msft-10k-2026.htm --value 155237
#   Operating Income Loss (2025-07-01..2026-06-30) = 1.55237e+11 U_USD
```

XBRL tags figures, not prose, so risk factors and MD&A narrative still come from
the HTML loader — the two are complementary.

On failure the pipeline retries once with the specific problems named, which
converts most near-misses into a correct citation. If it still fails, the answer
is withheld and the caller gets the verification report instead:

```json
{
  "answerable": false,
  "answer": "",
  "refusal_reason": "Answer withheld by the citation guardrail - figures not found in the filing excerpts: 9,999.",
  "verification": { "passed": false, "ungrounded_numbers": ["9,999"] }
}
```

Both blocking rules are configurable (`GuardrailConfig`) if you would rather
report than withhold.

## Architecture

Every stage is an ABC with a factory, so a component swaps by constructor
argument and nothing downstream changes. `SecFilingRagPipeline` is the only
object an application holds.

```
              ┌─────────────────────────────────────────────┐
  filings ──► │ DocumentLoader        (PDF / EDGAR HTML / txt)│
              │   └─ LoaderRegistry: suffix dispatch          │
              ├─────────────────────────────────────────────┤
              │ Chunker               TableAwareChunker       │
              ├─────────────────────────────────────────────┤
              │ EmbeddingModel        SentenceTransformer     │
              │                       Hashing (no deps)       │
              ├─────────────────────────────────────────────┤
              │ VectorStore           InMemory / Chroma       │
              ├─────────────────────────────────────────────┤
  question ─► │ Retriever             Hybrid (dense + BM25)   │
              │                       RRF fusion, table bias  │
              ├─────────────────────────────────────────────┤
              │ AnswerGenerator       Claude (structured out) │
              ├─────────────────────────────────────────────┤
              │ CitationVerifier      verbatim + numeric      │
              └──────────────────┬──────────────────────────┘
                                 ▼
                     AnswerResult (answer + claims + citations
                                   + verification report)
```

```python
from sec_copilot import SecFilingRagPipeline, Settings
from sec_copilot.vectorstore.chroma import ChromaVectorStore

pipeline = SecFilingRagPipeline(
    Settings(),
    store=ChromaVectorStore(dimension=384, persist_directory="index"),
)
pipeline.ingest("filings/aapl-10k-2024.pdf", ticker="AAPL", form_type="10-K", fiscal_year=2024)
result = pipeline.answer("How did gross margin move year over year?")
```

### Retrieval

Dense and BM25 run in parallel and are fused by **reciprocal rank** rather than
score averaging - cosine in `[-1, 1]` and unbounded BM25 are not comparable, and
per-query normalisation is unstable when one side returns few candidates.

Sparse retrieval is not optional here. Analysts search for exact strings -
`Item 1A`, `goodwill impairment`, `18,742` - and a dense encoder that has never
seen a given figure will happily return a semantically similar paragraph
containing the wrong number. The tokenizer keeps `394,328` and `1.5` intact for
the same reason: in a filing, the figure *is* the search term.

Numeric-looking questions get a configurable score boost on table chunks, since
that is where the answer lives.

An optional cross-encoder reranker (`retrieval/reranker.py`) reorders a deeper
shortlist. It is **off by default** — see the measurements below for why.

### Service concerns

`service.py` carries API-key auth (`X-API-Key`, constant-time comparison, off
when unconfigured so local runs need no setup), a per-caller sliding-log rate
limiter (a fixed window would let a caller fire two full quotas either side of a
boundary — worth preventing on an endpoint that spends money), and a usage
tracker that reports tokens, cache-hit ratio and an estimated cost at `/usage`.
Routes are `def`, not `async def`, so Starlette runs the blocking model call in
a worker thread instead of stalling the event loop.

### No framework

There is no LangChain, LangGraph, LlamaIndex or Unstructured here, deliberately.
The table problem *is* the project, and a general-purpose splitter is the thing
that breaks it; the three parsing defects fixed against real filings (ragged
currency columns, negatives split across cells, headings fragmented by inline
markup) are ones you only find by owning the parser. There is also no
orchestration to frame — the flow is one linear pass: retrieve → generate →
verify → retry once. BM25 and the vector store are about 150 lines each.
`LoaderRegistry.register()` is the seam if you do want a third-party loader.

### Providers

The model that writes the answer is a config switch; the guardrails do not know
or care which one produced it.

```bash
# Free tier, no card required - get a key at https://aistudio.google.com/apikey
echo 'GEMINI_API_KEY=your-key' > .env          # .env is gitignored
python -m sec_copilot.cli --provider gemini ask "What was total net revenue in 2025?"

python -m sec_copilot.cli --provider claude ask "..."   # needs ANTHROPIC_API_KEY
python -m sec_copilot.cli ask --offline "..."           # no key at all, echo stub
```

`SEC_COPILOT_PROVIDER=claude|gemini|echo` sets the default.

Two things the Gemini path has to work around, both handled in
`generation/gemini_client.py`:

* **Schema flattening.** Gemini's `responseSchema` is an OpenAPI subset that
  rejects the `$ref`/`$defs` pydantic emits for nested models and silently drops
  length keywords. `to_gemini_schema()` inlines the references, so
  `GroundedAnswer` stays the only representable response; and because
  `maxLength` cannot be enforced at generation time, an over-long quote is
  truncated rather than failing validation and discarding a correct answer.
* **Model fallback.** Free-tier capacity is per model and moves around — during
  development `gemini-flash-latest` and `gemini-3.8-flash` both returned a
  persistent 503 while three other models answered the identical payload in
  seconds. The client walks a candidate list rather than retrying one model
  harder, and reports which model actually answered.

### Model configuration

Claude Opus 5 with adaptive thinking and `effort: high` — reconciling figures
across statements is exactly the work that degrades without it. The system
prompt carries a cache breakpoint, so repeated questions against the same corpus
pay for the guardrail preamble once. Override with `--model` or
`SEC_COPILOT_MODEL`.

## Configuration

Environment variables, or a `Settings` object:

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required only when the provider is `claude` |
| `GEMINI_API_KEY` | — | Required only when the provider is `gemini`; read from `.env` too |
| `SEC_COPILOT_PROVIDER` | `claude` | `claude` \| `gemini` \| `echo` |
| `SEC_COPILOT_API_KEYS` | — | Comma-separated keys for `X-API-Key`. Unset = unauthenticated |
| `SEC_COPILOT_RATE_LIMIT` | `60` | Requests per window, per caller. `0` disables |
| `SEC_COPILOT_RATE_WINDOW` | `60` | Rate-limit window, seconds |
| `SEC_COPILOT_INDEX_DIR` | `index` | Where the index is persisted |
| `SEC_COPILOT_EMBEDDING_BACKEND` | `auto` | `auto` \| `sentence-transformers` \| `hashing` |
| `SEC_COPILOT_EMBEDDING_MODEL` | `all-mpnet-base-v2` | Chosen by measurement — see the table above |
| `SEC_COPILOT_VECTOR_STORE` | `in-memory` | `in-memory` \| `chroma` |
| `SEC_COPILOT_MODEL` | `claude-opus-5` | |
| `SEC_COPILOT_EFFORT` | `high` | `low` … `max` |

An index records the embedding model that built it, and loading it under a
different one fails loudly — querying across embedding spaces returns
plausible-looking neighbours that are meaningless.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

161 tests, no network and no API key required. They cover table integrity under
chunking, EDGAR column alignment, inline-XBRL fact extraction, page attribution,
hybrid retrieval, reranking, index round-tripping, auth/rate-limiting/usage, the
HTTP surface, and — the important ones — that a fabricated snippet, a citation
to an unretrieved chunk, a wrong page, an invented figure, and **a real figure
attributed to the wrong fiscal year** are each caught and blocked.

Several tests assert that parsed statements **reconcile arithmetically**
(gross margin − opex = operating income; current + non-current assets = total
assets). That is the strongest cheap check available: a silently misaligned
column fails it even though the rendered table still looks well-formed.

To include the real-filing integration tests:

```bash
python -m evals.fetch_filings   # Apple FY24/FY25 and Microsoft 10-Ks from EDGAR
python -m pytest -q             # test_real_filings.py stops skipping
```

## Measured retrieval quality

Run against three real 10-Ks (Apple FY2025 and FY2024, Microsoft — 970 chunks):

```bash
python -m evals.fetch_filings
python -m evals.retrieval_eval [--backend hashing] [--rerank]
```

The headline metric is **answerable@k — does the answer figure actually reach
the model's context?** That is the only retrieval property generation depends
on. Scoring against one designated "gold chunk" understates quality badly here,
because a figure legitimately appears in the statement, the MD&A discussion and
the segment note, and any of them lets the model answer.

16 single-figure questions, filtered by ticker/year (how an analyst queries):

| Embedding model | a@3 | a@8 | a@12 | MRR | ms/query |
|---|---|---|---|---|---|
| Hashing (no dependencies) | 0.69 | 0.81 | 0.81 | 0.560 | 3 |
| MiniLM-L6-v2 | 0.69 | 0.75 | 0.88 | 0.580 | 10 |
| bge-small-en-v1.5 | 0.94 | 1.00 | 1.00 | 0.783 | 17 |
| e5-base-v2 | 0.81 | 0.94 | 1.00 | 0.768 | 17 |
| **all-mpnet-base-v2 (default)** | **0.94** | **1.00** | **1.00** | **0.854** | **20** |

Plus 6 comparative and cross-filing questions, where *every* figure must arrive
together ("how did operating income change from FY2024 to FY2025", "which
company had higher operating income, Apple or Microsoft"): **complete@12 =
1.00**.

Four things worth reading off that:

- **The default encoder was chosen by measurement.** MiniLM — the obvious
  default, and what this started with — is the worst of the four tested;
  swapping to mpnet took answerable@8 from 0.75 to **1.00**, closing the
  retrieval gap on this corpus entirely. `bge-small` is the fast alternative:
  a@8 = 1.00 at a third of the size.
- **The hashing fallback is genuinely usable**, which matters because it needs
  no torch wheel: a@8 = 0.81 with BM25 carrying exact-figure lookups.
- **The cross-encoder reranker did not earn a default-on.** It improves coverage
  at k≥8 but is much worse at top-1 — an ms-marco model is trained on web prose
  and demotes dense numeric tables. It is available and off by default; with
  mpnet already at a@8 = 1.00 there is nothing left for it to recover.
- **Comparative questions now clear too** (1.00), which they did not before the
  loader learned to keep `September 27, 2025` from welding into one token — a
  reminder that parsing quality and retrieval quality are the same problem.

Chunk sizing was tuned against this harness too: smaller table chunks improved a
strict single-chunk metric but *reduced* answerability, so the defaults stayed.

## Layout

```
sec_copilot/
  models.py            domain models; the only shapes crossing boundaries
  config.py            frozen settings
  pipeline.py          the service facade
  api.py               FastAPI app
  cli.py               command line interface
  service.py           API-key auth, rate limiting, usage/cost tracking
  ingestion/           DocumentLoader ABC + PDF / EDGAR HTML / text loaders
                       + inline-XBRL fact extraction
  chunking/            Chunker ABC + TableAwareChunker
  embeddings/          EmbeddingModel ABC + sentence-transformers / hashing
  vectorstore/         VectorStore ABC + in-memory / Chroma
  retrieval/           Retriever ABC + hybrid RRF, BM25, cross-encoder rerank
  generation/          prompts, Claude client, CitationVerifier, period/scale logic
evals/                 EDGAR fetcher + answerable@k harness
sample_data/           a synthetic 10-K exercising every path
filings/               real 10-Ks (gitignored; fetched on demand)
tests/
```
