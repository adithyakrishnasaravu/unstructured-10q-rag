import json, os
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def load_and_chunk(json_dir="outputs/unstructured_json"):
    docs = []

    splitter = RecursiveCharacterTextSplitter( # long text into smaller chunks without cutting sentences
        chunk_size=500,
        chunk_overlap=100
    )

    # load and iterate over all JSON files post ETL
    for path in Path(json_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            elements = json.load(f)

        for e in elements:
            text = e.get("text")

            # skipping very short text elements
            if not text or len(text.strip()) < 30:
                continue
                
            metadata = e.get("metadata", {})
            metadata["source_file"] = path.name

            # Add company from filename
            fname = path.stem.lower()
            if "aapl" in fname or "apple" in fname:
                metadata["company"] = "Apple"
            elif "nvda" in fname or "nvidia" in fname:
                metadata["company"] = "NVIDIA"
            elif "goog" in fname or "google" in fname or "alphabet" in fname:
                metadata["company"] = "Alphabet"

            metadata = sanitize_metadata_for_pinecone(metadata)

            # Tables as atomic chunks, and little description
            if e.get("type") == "Table":
                docs.append(
                    Document(
                        page_content=f"""Financial table from {metadata.get('company')} 10-Q.
            {text}""".strip(),
                        metadata={**metadata, "element_type": "table"},
                    )
                )
            else:
                base_doc = Document(
                    page_content=text,
                    metadata={**metadata, "element_type": "text"},
                )
                docs.extend(splitter.split_documents([base_doc]))
        print(f"Loaded {len(elements)} elements from {path.name}")
    return docs


def embeddings_and_index(docs, index_name="financial-10q", namespace="financial-10q-q3"):
    # OpenAI embedding model (3072-dim for text-embedding-3-large)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Embed each chunk and insert vectors + metadata into Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
    return vectorstore


def sanitize_metadata_for_pinecone(meta: dict) -> dict:
    """Keep only Pinecone-supported metadata value types.
    This sanitizeizes the metadata to ensure it can be stored in Pinecone."""
    out = {}
    if not meta:
        return out

    for k, v in meta.items():
        if v is None:
            continue

        if isinstance(v, (str, int, float, bool)):
            out[k] = v
            continue

        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = v
            continue

        if k == "links" and isinstance(v, list):
            urls = []
            for item in v:
                if isinstance(item, dict) and item.get("url"):
                    urls.append(str(item["url"]))
            if urls:
                out["link_urls"] = urls  # store as list[str]
            continue
    return out

def main_():
    # 1) Load & chunk
    docs = load_and_chunk()
    print(f"Total chunks: {len(docs)}")

    # 2) Embeddings & index
    vectorstore = embeddings_and_index(docs)

    # 3) Query retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4) LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    # 5) System prompt with grounding instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a financial assistant with expertise in analyzing financial documents. Answer ONLY using the provided context. "
        "If the context is insufficient, say you don't know."
        "If units are explicitly stated (e.g., 'in millions'), you may convert to billions and show the conversion."),
        ("human",
        "Question: {input}\n\n"
        "Context:\n{context}\n\n"
        "Answer:")
    ])

    # 6) Query
    query = "What is the cash and cash equivalents of Alphabet at the end of Q3?"

    # Pass the query and retrieve top 5 most relevant chunks
    retrieved_chunks = retriever.invoke(query)

    # Combine retrieved chunks into a single context string
    context = "\n\n---\n\n".join(d.page_content for d in retrieved_chunks)

    # Generate answer
    messages = prompt.format_messages(input=query, context=context)
    answer = llm.invoke(messages)

    # 7) Output
    print("\nANSWER:\n", answer.content)

    print("\nSOURCES:")
    for d in retrieved_chunks:
        print("-", d.metadata.get("company"), d.metadata.get("source_file"), "p.", d.metadata.get("page_number"))


if __name__ == "__main__":
    main_()