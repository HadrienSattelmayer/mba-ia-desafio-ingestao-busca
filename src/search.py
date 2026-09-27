import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

REQUIRED_VARS = [
    "GOOGLE_API_KEY",
    "GOOGLE_EMBEDDING_MODEL",
    "GOOGLE_LLM_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
]

TOP_K = 10


def build_chain(store, llm):
    """Monta a chain: pergunta -> busca vetorial (k=10) -> prompt -> LLM -> texto."""

    def retrieve_context(question: str) -> str:
        results = store.similarity_search_with_score(question, k=TOP_K)
        return "\n\n".join(doc.page_content for doc, _score in results)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    return (
        {
            "contexto": RunnableLambda(retrieve_context),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def search_prompt(question=None):
    """
    Retorna a chain de busca pronta para uso (chain.invoke("pergunta")).
    Se `question` for informada, já retorna a resposta.
    Retorna None se a inicialização falhar.
    """
    missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
    if missing:
        print(f"Variáveis de ambiente ausentes: {', '.join(missing)}")
        return None

    try:
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

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL"),
            temperature=0,
            transport="rest",
        )
    except Exception as e:
        print(f"Erro ao inicializar a busca: {e}")
        return None

    chain = build_chain(store, llm)

    if question:
        return chain.invoke(question)

    return chain