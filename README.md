# Ingestão e Busca Semântica com LangChain e Postgres

Sistema RAG que faz a ingestão de um PDF em um banco PostgreSQL com pgVector e permite fazer perguntas via linha de comando, com respostas baseadas **apenas** no conteúdo do documento.

## Arquitetura

```
Ingestão:  document.pdf → PyPDFLoader → RecursiveCharacterTextSplitter (1000 / 150)
                        → Gemini Embeddings → PGVector (PostgreSQL)

Busca:     pergunta → embedding → similarity_search_with_score (k=10)
                    → prompt com contexto → Gemini (LLM) → resposta
```

## Tecnologias

- Python 3.11+
- LangChain
- PostgreSQL 17 + pgVector (via Docker Compose)
- Google Gemini (embeddings e LLM)

## Estrutura

```
├── docker-compose.yml    # PostgreSQL + pgVector
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Ingestão do PDF no banco vetorial
│   ├── search.py         # Busca vetorial + prompt + LLM
│   ├── chat.py           # CLI de perguntas e respostas
├── document.pdf          # PDF para ingestão
└── README.md
```

## Pré-requisitos

- Python 3.11 ou superior
- Docker e Docker Compose
- Uma API Key do Google Gemini ([Google AI Studio](https://aistudio.google.com/app/apikey))

## Como executar

### 1. Clonar o repositório

```bash
git clone <url-deste-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar e ativar o ambiente virtual

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar as variáveis de ambiente

Copie o template e preencha a `GOOGLE_API_KEY`:

```bash
cp .env.example .env
```

| Variável | Descrição | Exemplo |
|---|---|---|
| `GOOGLE_API_KEY` | API Key do Google Gemini | `AIza...` |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings | `models/gemini-embedding-001` |
| `GOOGLE_LLM_MODEL` | Modelo de LLM para as respostas | `gemini-2.5-flash-lite` |
| `DATABASE_URL` | Conexão com o Postgres (driver psycopg 3) | `postgresql+psycopg://postgres:postgres@127.0.0.1:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector | `documentos` |
| `PDF_PATH` | Caminho do PDF a ser ingerido | `document.pdf` |

> Os nomes de modelos mudam com frequência. Confira os modelos disponíveis na [documentação oficial do Gemini](https://ai.google.dev/gemini-api/docs/models).

### 5. Subir o banco de dados

```bash
docker compose up -d
```

O serviço `bootstrap_vector_ext` cria automaticamente a extensão `vector` no banco.

### 6. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada:

```
Ingestão concluída: 34 páginas, 67 chunks armazenados.
```

A ingestão usa IDs determinísticos por chunk, então pode ser executada novamente sem duplicar os dados.

### 7. Rodar o chat

```bash
python src/chat.py
```

Digite `sair` (ou `Ctrl+C`) para encerrar.

## Exemplo de uso

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da empresa Alfa Energia S.A.?
RESPOSTA: O faturamento da Alfa Energia S.A. foi de R$ 722.875.391,46.

---

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

---
```

## Troubleshooting

- **Troca de modelo de embeddings:** modelos diferentes geram vetores de dimensões diferentes. Ao trocar de modelo, apague o volume do banco e refaça a ingestão:

  ```bash
  docker compose down -v
  docker compose up -d
  python src/ingest.py
  ```

- **`connection timeout` ao conectar no banco:** verifique se o container está saudável (`docker compose ps`) e se a porta 5432 não está ocupada por outra instalação local do PostgreSQL. Se estiver, altere o mapeamento de porta no `docker-compose.yml` (ex.: `"5433:5432"`) e ajuste a `DATABASE_URL`.

- **`DATABASE_URL` sem `+psycopg`:** o `langchain-postgres` usa o driver psycopg 3; a URL precisa começar com `postgresql+psycopg://`.