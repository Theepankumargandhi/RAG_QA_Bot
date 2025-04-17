import cohere
import fitz
import pinecone


class EmbeddingUtils:
    def __init__(self, pdf_path: str, cohere_api_key: str, pinecone_api_key: str):
        self.pdf_path = pdf_path
        self.co = cohere.Client(cohere_api_key)
        self.pinecone_api_key = pinecone_api_key
        self.chunks = []
        self.embeddings = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load_pdf()
        self.split_text()
        self.embed_chunks()
        self.index_chunks()

    def load_pdf(self):
        self.pdf_text = self.extract_text_from_pdf(self.pdf_path)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text("text")
        return text

    def split_text(self, chunk_size=1000):
        sentences = self.pdf_text.split(". ")
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                self.chunks.append(current_chunk)
                current_chunk = sentence + ". "
        if current_chunk:
            self.chunks.append(current_chunk)

    def embed_chunks(self, batch_size=90):
        total_chunks = len(self.chunks)
        for i in range(0, total_chunks, batch_size):
            batch = self.chunks[i:min(i + batch_size, total_chunks)]
            batch_embeddings = self.co.embed(
                texts=batch, input_type="search_document", model="embed-english-v3.0"
            ).embeddings
            self.embeddings.extend(batch_embeddings)

    def index_chunks(self):
        if not self.embeddings:
            raise ValueError("No embeddings found. Make sure embed_chunks() ran successfully.")

        # Initialize Pinecone
        pinecone.init(api_key=self.pinecone_api_key, environment="us-east-1")
        index_name = "rag-qa-bot"

        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=len(self.embeddings[0]),
                metric="cosine"
            )

        # Connect to the index
        self.index = pinecone.Index(index_name)

        # Prepare metadata and upsert vectors
        chunks_metadata = [{'text': chunk} for chunk in self.chunks]
        ids = [str(i) for i in range(len(self.chunks))]

        self.index.upsert(vectors=[
            {"id": idx, "values": vector, "metadata": metadata}
            for idx, vector, metadata in zip(ids, self.embeddings, chunks_metadata)
        ])

    def retrieve(self, query: str) -> list:
        query_emb = self.co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]  # single embedding

        res = self.index.query(
            vector=query_emb,
            top_k=self.retrieve_top_k,
            include_metadata=True
        )

        if not res.get('matches'):
            print("No matches found in Pinecone for this query.")
            return []

        docs_to_rerank = [match['metadata']['text'] for match in res['matches']]

        if not docs_to_rerank:
            print("No documents available to rerank.")
            return []

        rerank_results = self.co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v2.0"
        )

        return [res['matches'][result.index]['metadata'] for result in rerank_results.results]
