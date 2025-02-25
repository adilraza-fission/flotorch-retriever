from reader.json_reader import JSONReader
from embedding.embedding import BaseEmbedding
from chunking.chunking import Chunk
from rerank.rerank import BedrockReranker
from storage.db.vector.vector_storage import VectorStorage
from pydantic import BaseModel
from inferencer.inferencer import BaseInferencer

class Question(BaseModel):
    question: str
    answer: str

    def get_chunk(self) -> Chunk:
        return Chunk(data=self.question)

class Retriever:
    def __init__(self, json_reader: JSONReader, embedding: BaseEmbedding, vector_storage: VectorStorage, inferencer: BaseInferencer, reranker: BedrockReranker) -> None:
        self.json_reader = json_reader
        self.embedding = embedding
        self.vector_storage = vector_storage
        self.inferencer = inferencer
        self.reranker = reranker

    def retrieve(self, path: str, query: str, knn: int, hierarchical: bool = False):
       
        questions_list = self.json_reader.read_as_model(path, Question)
        for question in questions_list:
            question_chunk = question.get_chunk()
            response = self.vector_storage.search(question_chunk, knn, hierarchical)
            if response.status:
                vector_response = response.to_json()['result']
                vector_response = self.reranker.rerank_documents(question_chunk.data, vector_response)
                metadata, answer = self.inferencer.generate_text(question.question, vector_response)
            else:
                answer = response.metadata['guardrail_output']
            
            