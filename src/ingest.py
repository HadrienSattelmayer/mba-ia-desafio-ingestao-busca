import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

REQUIRED_VARS = [
    "GOOGLE_API_KEY",
    "GOOGLE_EMBEDDING_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
    "PDF_PATH",
]

# Lotes pequenos com pausa entre eles para respeitar o limite por minuto do free tier
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "10"))
PAUSE_BETWEEN_BATCHES = float(os.getenv("INGEST_PAUSE_SECONDS", "10"))
MAX_RETRIES = 5


def is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "quota" in message or "resource_exhausted" in message


def add_batch_with_retry(store, batch, batch_ids):
    """Envia um lote ao PGVector; em erro de cota (429), espera com backoff e tenta de novo."""
    wait = 30
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            store.add_documents(documents=batch, ids=batch_ids)
            return
        except Exception as e:
            if not is_quota_error(e) or attempt == MAX_RETRIES:
                raise
            print(f"  Limite da API atingido (tentativa {attempt}/{MAX_RETRIES}). Aguardando {wait}s...")
            time.sleep(wait)
            wait *= 2


def ingest_pdf():
    missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Variáveis de ambiente ausentes: {', '.join(missing)}")

    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF não encontrado: {PDF_PATH}")

    docs = PyPDFLoader(PDF_PATH).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise RuntimeError("Nenhum texto foi extraído do PDF.")

    # Mantém só os metadados úteis e descarta valores vazios
    for chunk in chunks:
        chunk.metadata = {
            key: value
            for key, value in chunk.metadata.items()
            if key in ("source", "page") and value not in ("", None)
        }

    # transport="rest" evita o cliente gRPC assíncrono, que gera erro ruidoso ao encerrar o Python
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
        transport="rest",
    )

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    # IDs determinísticos: permitem retomar uma ingestão interrompida sem duplicar chunks
    ids = [f"doc-{i}" for i in range(len(chunks))]

    existing_ids = {doc.id for doc in store.get_by_ids(ids)}
    pending = [(chunk, chunk_id) for chunk, chunk_id in zip(chunks, ids) if chunk_id not in existing_ids]

    if existing_ids:
        print(f"{len(existing_ids)} chunks já estavam no banco e serão ignorados.")

    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    for n, start in enumerate(range(0, len(pending), BATCH_SIZE), start=1):
        batch = pending[start:start + BATCH_SIZE]
        add_batch_with_retry(store, [c for c, _ in batch], [i for _, i in batch])
        print(f"  Lote {n}/{total_batches} armazenado ({len(batch)} chunks).")

        if n < total_batches:
            time.sleep(PAUSE_BETWEEN_BATCHES)

    print(f"Ingestão concluída: {len(docs)} páginas, {len(chunks)} chunks no banco.")


if __name__ == "__main__":
    ingest_pdf()