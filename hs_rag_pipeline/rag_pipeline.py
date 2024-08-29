import argparse
from pathlib import Path
from haystack import Pipeline
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# Dictionary to map vector DB types to their respective initializer functions
DOCUMENT_STORE_INITIALIZERS = {}

def register_document_store(store_type):
    def wrapper(func):
        DOCUMENT_STORE_INITIALIZERS[store_type.lower()] = func
        return func
    return wrapper

@register_document_store("pinecone")
def init_pinecone_document_store(index_dimension, **kwargs):
    api_key = Secret.from_token(kwargs.get("api_key"))
    return PineconeDocumentStore(
        api_key=api_key,
        dimension=index_dimension,
        index=kwargs.get("index_name"),
    )

def get_document_store(store_type, index_dimension, **kwargs):
    try:
        # Dynamically call the initializer function based on the store type
        init_func = DOCUMENT_STORE_INITIALIZERS[store_type.lower()]
        return init_func(index_dimension=index_dimension, **kwargs)
    except KeyError:
        raise ValueError(f"Unsupported vector DB type: {store_type}")

def get_retriever(document_store):
    return PineconeEmbeddingRetriever(
        document_store=document_store,
    )

def main(args):
    document_store = get_document_store(
        args.vector_db_type,
        args.index_dimension,
        api_key=args.pinecone_api_key,
        index_name=args.pinecone_index_name,
    )

    # Create a pipeline for preprocessing
    preprocess_pipeline = Pipeline()

    preprocess_pipeline.add_component(instance=TextFileToDocument(), name="text_file_converter")
    preprocess_pipeline.add_component(instance=DocumentCleaner(), name="cleaner")
    preprocess_pipeline.add_component(instance=DocumentSplitter(split_by=args.preprocess_split_by, split_length=args.preprocess_split_length), name="splitter")
    preprocess_pipeline.connect("text_file_converter.documents", "cleaner.documents")
    preprocess_pipeline.connect("cleaner.documents", "splitter.documents")

    path = Path(args.data_path)
    files = list(path.glob(f"*.{args.file_format}"))
    preprocess_result = preprocess_pipeline.run({"text_file_converter": {"sources": files}})

    docs = preprocess_result["splitter"]["documents"]

    if not docs:
        print("No documents to embed. Make sure data_path is correct and contains files with the specified file_format.")
        return

    print(f"Number of documents after preprocessing: {len(docs)}")
    print(f"Using embedding model: {args.embedding_model}")

    # Create a pipeline for indexing
    indexing_pipeline = Pipeline()

    # SentenceTransformersTextEmbedder can be retrieved like get_document_store
    indexing_pipeline.add_component(SentenceTransformersDocumentEmbedder(model=args.embedding_model), name="embedder")
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    indexing_pipeline.connect("embedder", "writer")

    indexing_pipeline.run({"documents": docs})
    print("Documents have been successfully indexed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vectorize and index records in a vector database using Haystack 2.0."
    )

    # General arguments

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing the data files to be indexed",
    )

    parser.add_argument(
        "--file_format",
        type=str,
        required=True,
        help="Type of the files (txt)",
    )

    # Preprocessing arguments

    parser.add_argument(
        "--preprocess_split_by",
        type=str,
        default="word",
        help="Strategy to split the document",
    )

    parser.add_argument(
        "--preprocess_split_length",
        type=int,
        default=100,
        help="Length to split the document",
    )

    # Vectorization arguments

    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="The embedding model to use for vectorization, example: 'all-MiniLM-L6-v2' when using 'sentence-transformers'",
    )

    # Indexing arguments

    parser.add_argument(
        "--vector_db_type",
        type=str,
        required=True,
        help="Type of vector database (pinecone)",
    )

    parser.add_argument(
        "--index_dimension",
        type=int,
        required=True,
        help="Embedding dimension for the vector database",
    )

    # Indexing: Pinecone-specific arguments
    parser.add_argument("--pinecone_api_key", type=str, help="API key for Pinecone")
    parser.add_argument(
        "--pinecone_index_name", type=str, help="Index name for Pinecone"
    )

    args = parser.parse_args()
    main(args)
