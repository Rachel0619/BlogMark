from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.indices.base import IndexType
from llama_index.node_parser.topic import TopicNodeParser
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv, find_dotenv
from trafilatura import fetch_url, extract, extract_metadata
from qdrant_client.models import Filter, FieldCondition, MatchValue
import qdrant_client
import pandas as pd
import hashlib
import os


class BlogRAGHandler:

    def __init__(self, chunk_size: int, chunk_overlap: int, top_k: int, urls_file: str = 'urls.csv'):

        load_dotenv(find_dotenv())

        self.urls_file = urls_file
        collection_name = os.environ.get('COLLECTION_NAME')
        qdrant_url = os.environ.get('QDRANT_URL')
        # qdrant_api_key = os.environ.get('QDRANT_API_KEY')
        llm_url = os.environ.get('LLM_URL')
        llm_model = os.environ.get('LLM_MODEL')
        embed_model_name = os.environ.get('EMBED_MODEL_NAME')

        # Initialize settings
        self.top_k = top_k
        self.collection_name = collection_name

        # Setting up LLM and embedding model
        Settings.llm = Ollama(base_url=llm_url, model=llm_model, request_timeout=300)
        Settings.embed_model = OllamaEmbedding(base_url=llm_url, model_name=embed_model_name)
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        # Initialize Qdrant client BEFORE loading documents
        self.qdrant_client = qdrant_client.QdrantClient(url=qdrant_url)

        # Load blog documents from URLs (now qdrant_client is available)
        self.documents = self.load_blog_documents()

        # Set up Qdrant vector store
        self.qdrant_vector_store = QdrantVectorStore(collection_name=self.collection_name,
                                                     client=self.qdrant_client,
                                                     fastembed_sparse_model=os.environ.get("sparse_model"),
                                                     enable_hybrid=True
                                                     )

        # Set up StorageContext
        self.storage_ctx = StorageContext.from_defaults(vector_store=self.qdrant_vector_store)

        self.vector_store_index: IndexType = None

        self.sentence_splitter = SentenceSplitter.from_defaults(chunk_size=Settings.chunk_size,
                                                                chunk_overlap=Settings.chunk_overlap)

    def scrape_blog_simple(self, url) -> dict:
        """
        Extract content and metadata from a blog post URL using trafilatura.
        """
        downloaded = fetch_url(url)
        if not downloaded:
            print(f"Failed to download: {url}")
            return None

        text = extract(downloaded)
        if not text:
            print(f"Failed to extract text content: {url}")
            return None

        metadata = extract_metadata(downloaded)

        content = {
            'url': url,
            'text': text,
            'title': getattr(metadata, 'title', None) if metadata else None,
            'date': getattr(metadata, 'date', None) if metadata else None,
            'author': getattr(metadata, 'author', None) if metadata else None,
            'sitename': getattr(metadata, 'sitename', None) if metadata else None
        }
        
        return content

    def url_exists_in_index(self, url):
        """Check if URL already exists in the index"""
        if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
            return False
        
        # Simple check using Qdrant's filter
        
        search_result = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="url", match=MatchValue(value=url))]
            ),
            limit=1
        )
        
        return len(search_result[0]) > 0

    def load_blog_documents(self):
        """
        Load blog URLs from CSV and scrape content to create LlamaIndex Documents.
        Only processes URLs that haven't been processed before.
        """
        urls = pd.read_csv(self.urls_file, header=None, names=['link'])
        
        new_urls = []
        for url in urls['link']:
            if not self.url_exists_in_index(url):
                new_urls.append(url)
        
        if not new_urls:
            print("No new URLs to process. All URLs already exist in the database.")
            return []
        
        print(f"Processing {len(new_urls)} new URLs")
        
        scraped_blogs = []
        for url in new_urls:
            blog_data = self.scrape_blog_simple(url)
            if blog_data:
                scraped_blogs.append(blog_data)
        
        # Convert to LlamaIndex Documents
        documents = []
        for blog_data in scraped_blogs:
            if blog_data and blog_data.get('text'):
                # Create metadata dict
                metadata = {
                    'title': blog_data.get('title') or 'No Title',
                    'date': blog_data.get('date') or 'Unknown',
                    'author': blog_data.get('author') or 'Unknown',
                    'sitename': blog_data.get('sitename') or 'Unknown',
                    'url': blog_data.get('url'),
                    'content_length': len(blog_data['text'])
                }
                
                # Create Document with text and metadata
                doc = Document(
                    text=blog_data['text'],
                    metadata=metadata
                )
                documents.append(doc)
        
        print(f"Successfully loaded {len(documents)} new blog documents")
        return documents

    def index_data_based_on_method(self, method: str):

        if method == 'semantic_chunking':
            # Initialize splitters
            splitter = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
            )
            if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
                # Create VectorStoreIndex
                self.vector_store_index = VectorStoreIndex(
                    nodes=splitter.get_nodes_from_documents(documents=self.documents),
                    storage_context=self.storage_ctx, show_progress=True, transformations=[self.sentence_splitter],
                    embed_model=OllamaEmbedding(model_name=os.environ.get("EMBED_MODEL_NAME"))
                )
            else:
                self.vector_store_index = VectorStoreIndex.from_vector_store(vector_store=self.qdrant_vector_store,
                                                                             embed_model=OllamaEmbedding(model_name=os.environ.get("EMBED_MODEL_NAME")))

        elif method == 'semantic_double_merge_chunking':
            config = LanguageConfig(language="english", spacy_model="en_core_web_md")
            splitter = SemanticDoubleMergingSplitterNodeParser(
                language_config=config,
                initial_threshold=0.4,
                appending_threshold=0.5,
                merging_threshold=0.5,
                max_chunk_size=5000,
            )
            if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
                # Create VectorStoreIndex
                self.vector_store_index = VectorStoreIndex(
                    nodes=splitter.get_nodes_from_documents(documents=self.documents),
                    storage_context=self.storage_ctx, show_progress=True, transformations=[self.sentence_splitter],
                    embed_model=OllamaEmbedding(model_name=os.environ.get("EMBED_MODEL_NAME"))
                )
            else:
                self.vector_store_index = VectorStoreIndex.from_vector_store(vector_store=self.qdrant_vector_store,
                                                                             embed_model=OllamaEmbedding(model_name=os.environ.get("EMBED_MODEL_NAME")))

        elif method == 'topic_node_parser':
            node_parser = TopicNodeParser.from_defaults(
                llm=Settings.llm,
                max_chunk_size=Settings.chunk_size,
                similarity_method="llm",
                similarity_threshold=0.8,
                window_size=3  # paper suggests it as 5
            )
            if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
                # Create VectorStoreIndex
                self.vector_store_index = VectorStoreIndex(
                    nodes=node_parser.get_nodes_from_documents(documents=self.documents),
                    storage_context=self.storage_ctx, show_progress=True, transformations=[self.sentence_splitter],
                    embed_model=OllamaEmbedding(model_name=os.environ.get("EMBED_MODEL_NAME"))
                )
            else:
                self.vector_store_index = VectorStoreIndex.from_vector_store(vector_store=self.qdrant_vector_store,
                                                                             embed_model=OllamaEmbedding(model_name=os.environ.get("EMBED_MODEL_NAME")))

    def load_existing_index(self):
        """Load existing index from Qdrant if collection exists"""
        if self.qdrant_client.collection_exists(collection_name=self.collection_name):
            self.vector_store_index = VectorStoreIndex.from_vector_store(
                vector_store=self.qdrant_vector_store,
                embed_model=OllamaEmbedding(model_name=os.environ.get("EMBED_MODEL_NAME"))
            )
        else:
            raise ValueError(f"Collection '{self.collection_name}' does not exist. Run indexing first.")
    
    def create_query_engine(self):
        if self.vector_store_index is None:
            self.load_existing_index()
        return self.vector_store_index.as_query_engine(top_k=self.top_k)

    def query(self, query_text):
        query_engine = self.create_query_engine()
        return query_engine.query(str_or_query_bundle=query_text)


# Usage example
def main():
    blog_rag_handler = BlogRAGHandler(chunk_size=128, chunk_overlap=20, top_k=3, urls_file='urls.csv')
    blog_rag_handler.index_data_based_on_method(method='semantic_double_merge_chunking')
    # response = blog_rag_handler.query("what is tiny agent?")
    # print(response)

if __name__ == "__main__":
    main()
