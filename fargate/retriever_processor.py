from datetime import datetime
import json
from typing import Dict, List, Optional, Union
import uuid
from embedding.embedding_registry import embedding_registry
from embedding.guardrails.guardrails_embedding import GuardrailsEmbedding
from fargate.base_task_processor import BaseFargateTaskProcessor
from guardrails.guardrails import BedrockGuardrail
from inferencer.guardrails.guardrails_inferencer import GuardRailsInferencer
from inferencer.inferencer_provider_factory import InferencerProviderFactory
from logger.global_logger import get_logger
from config.config import Config
from config.env_config_provider import EnvConfigProvider
from reader.json_reader import JSONReader
from rerank.rerank import BedrockReranker
from retriever.retriever import Retriever
from storage.db.dynamodb import DynamoDB
from storage.db.vector.guardrails_vector_storage import GuardRailsVectorStorage
from storage.db.vector.open_search import OpenSearchClient
from inferencer.bedrock_inferencer import BedrockInferencer
from storage.db.vector.vector_storage_factory import VectorStorageFactory
from storage.storage_provider_factory import StorageProviderFactory

from embedding.titanv2_embedding import TitanV2Embedding
from embedding.titanv1_embedding import TitanV1Embedding
from embedding.cohere_embedding import CohereEmbedding
from embedding.bge_large_embedding import BGELargeEmbedding, BGEM3Embedding, GTEQwen2Embedding


logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)

class RetrieverProcessor(BaseFargateTaskProcessor):
    """
    Processor for retriever tasks in Fargate.
    """

    def process(self):
        logger.info("Starting retriever process.")
        try:
            #logic
            exp_config_data = {
                "ClusterArn": "arn:aws:ecs:us-east-1:677276078734:cluster/FlotorchCluster-devint",
                "IndexingTaskDefinitionArn": "arn:aws:ecs:us-east-1:677276078734:task-definition/FlotorchTaskIndexing-devint:2",
                "RetrieverTaskDefinitionArn": "arn:aws:ecs:us-east-1:677276078734:task-definition/FlotorchTaskRetriever-devint:2",
                "EvaluationTaskDefinitionArn": "arn:aws:ecs:us-east-1:677276078734:task-definition/FlotorchTaskEvaluation-devint:2",
                "SageMakerRoleArn": "arn:aws:iam::677276078734:role/flotorch-bedrock-role-devint",
                "temp_retrieval_llm": "0.1",
                "gt_data": "file://C:/Projects/files/gt.json",
                "vector_dimension": "1024",
                "eval_retrieval_model": "amazon.titan-text-express-v1",
                "chunk_size": "512",
                "eval_embedding_model": "amazon.titan-embed-text-v1",
                "experiment_id": "OVAP0D4F",
                "rerank_model_id": "none",
                "n_shot_prompts": "0",
                "embedding_service": "bedrock",
                "embedding_model": "amazon.titan-embed-image-v1",
                "bedrock_knowledge_base": False,
                "kb_data": "file://C:/Projects/files/medical_abstracts_100_169kb.pdf",
                "retrieval_service": "bedrock",
                "chunking_strategy": "fixed",
                "execution_id": "KFE8A",
                "aws_region": "us-east-1",
                "eval_service": "ragas",
                "knn_num": "3",
                "id": "OVAP0D4F",
                "retrieval_model": "amazon.titan-text-express-v1",
                "index_id": "local-index-1024",
                "chunk_overlap": "10",
                "indexing_algorithm": "hnsw",
                "enable_guardrails": True,
                "enable_prompt_guardrails": True,
                "enable_context_guardrails": True,
                "enable_response_guardrails": True,
                "guardrail_id": "71e2tdhnpolw",
                "guardrail_version": "DRAFT",
                # "hierarchical_parent_chunk_size": 512,
                # "hierarchical_child_chunk_size": 256,
                # "hierarchical_chunk_overlap_percentage": 10,
                "n_shot_prompt_guide_obj": {
                    "examples": [
                        {
                        "example": "Thrombotic thrombocytopenic purpura treated with high-dose intravenous gamma globulin. Plasma infusion and/or plasma exchange has become standard therapy in the treatment of thrombotic thrombocytopenic purpura (TTP). The management of patients in whom such primary therapy fails is difficult and uncertain. We have described a patient who obtained a sustained remission with the use of high-dose IV gamma globulin after an initial response to aggressive plasma exchange was followed by prompt relapse. Our case and others suggest that high-dose IV IgG may induce remission in patients with TTP who do not respond to standard plasma infusion and/or exchange. Answer: General pathological conditions"
                        },
                        {
                        "example": "Further notes on Munchausen's syndrome: a case report of a change from acute abdominal to neurological type. A rare case of Munchausen's syndrome beginning in early childhood is described. The diagnosis of Munchausen's syndrome was made at the age of 29 years, after the symptoms had changed from acute abdominal to neurological complaints, with feigned loss of consciousness, first ascribed to an encephalitis. Insight into the psychopathology of this patient is given by his biography, by assessment of a psychotherapist, who had treated him some years before, and by his observed profile in some psychological tests. Answer: Nervous system diseases"
                        },
                        {
                        "example": "Syndromes of transient amnesia: towards a classification. A study of 153 cases. Of 153 patients presenting with acute transient amnesia, 114 fulfilled the proposed strict diagnostic criteria for transient global amnesia (TGA). The prognosis of this group was excellent with the exception of a small subgroup (7%), largely identifiable because of atypically brief or recurrent attacks, who developed epilepsy of temporal lobe type on follow up. Computerised tomography (CT) scans performed on 95 patients were normal, evidence for covert alcoholism was lacking and there was a familial incidence of approximately 2%. By contrast, the group of 39 patients who did not meet the criteria for TGA had a significantly worse prognosis with a high incidence of major vascular events. The groups could not be distinguished on the basis of behavioural characteristics during the attack. The following classification was proposed: 1) pure TGA--attacks fulfilling the strict criteria, and of more than one hour in duration which do not require detailed investigation, 2) probable epileptic amnesia--attacks of less than an hour or rapidly recurrent, 3) probable transient ischaemic amnesia, a minority of cases with additional focal neurological deficits during the attack. Answer: Cardiovascular system diseases"
                        },
                        {
                        "example": "The value of intubating and paralyzing patients with suspected head injury in the emergency department. One hundred consecutive trauma patients who underwent planned emergency intubation with muscle paralysis in the ED were studied to investigate the safety of these procedures and to determine their impact on the evaluation of patients with suspected head injury. Patients were intubated by either a surgeon (n = 47) or anesthesiologist (n = 53), and paralyzed with either vecuronium (n = 59) or succinylcholine (n = 41). Nasal intubation was used in 40 patients, oral in 57, and cricothyroidotomy in three. Ninety-four patients with suspected head injuries had a CT scan performed. Fifty-five (59%) had a positive scan and 15 required emergent neurosurgical intervention. Only two patients had lateral cervical spine roentgenograms before intubation; seven patients were eventually found to have cervical fractures. No patient suffered a neurologic deficit. One patient developed aspiration pneumonia following intubation. The three failed intubations occurred in patients with multiple facial fractures. We conclude that induced paralysis and intubation in the ED is safe, can facilitate the diagnostic workup, and may be a potentially life-saving maneuver in combative trauma patients. Answer: Nervous system diseases"
                        }
                    ],
                    "system_prompt": " You are an expert medical practitioner. Read the attached knowledgebase and classify the following medical diagnosis into one of the 5 known diseases specifically neoplasms, digestive system diseases, nervous system diseases, cardiovascular diseases, and general pathological conditions. Your output should include 3 parameters.\n1. disease : Make sure you identify disease \nonly based on medical diagnosis provided below.\n2. context : Explain why you think this disease is most likely given the medical diagnosis \n3. confidence : Output a number between 0 and 100 based on how confident you are in disease identification from the given diagnosis.",
                    "user_prompt": "Now categorize the following medical cases into one of the 5 categories provided."
                }
            }

            logger.info(f"Into retriever processor. Processing event: {json.dumps(exp_config_data)}")

            gt_data = exp_config_data.get("gt_data")
            storage = StorageProviderFactory.create_storage_provider(gt_data)
            gt_data_path = storage.get_path(gt_data)
            json_reader = JSONReader(storage)

            embedding_class = embedding_registry.get_model(exp_config_data.get("embedding_model"))
            embedding = embedding_class(
                exp_config_data.get("embedding_model"), 
                exp_config_data.get("aws_region"), 
                int(exp_config_data.get("vector_dimension")))
            
            if exp_config_data.get("enable_guardrails", False):
                base_guardrails = BedrockGuardrail(exp_config_data.get("guardrail_id", ""), exp_config_data.get("guardrail_version", 0))
            
            
            vector_storage = VectorStorageFactory.create_vector_storage(
                use_bedrock_kb=exp_config_data.get("bedrock_knowledge_base", False),
                embedding=embedding,
                opensearch_host=config.get_opensearch_host(),
                opensearch_port=config.get_opensearch_port(),
                opensearch_username=config.get_opensearch_username(),
                opensearch_password=config.get_opensearch_password(),
                index_id=exp_config_data.get("index_id"),
                knowledge_base_id=exp_config_data.get("kb_data"),
                aws_region=exp_config_data.get("aws_region")
            )

            if exp_config_data.get("enable_guardrails", False) and exp_config_data.get("enable_prompt_guardrails", False):
                vector_storage = GuardRailsVectorStorage(
                    vector_storage, 
                    base_guardrails,
                    exp_config_data.get("enable_prompt_guardrails", False),
                    exp_config_data.get("enable_context_guardrails", False)
                )

            reranker = BedrockReranker(exp_config_data.get("aws_region"), exp_config_data.get("rerank_model_id")) \
                if exp_config_data.get("rerank_model_id").lower() != "none" \
                else None
            
            inferencer = InferencerProviderFactory.create_inferencer_provider(
                exp_config_data.get("retrieval_service"),
                exp_config_data.get("retrieval_model"), 
                exp_config_data.get("aws_region"), 
                int(exp_config_data.get("n_shot_prompts")), 
                float(exp_config_data.get("temp_retrieval_llm")), 
                exp_config_data.get("n_shot_prompt_guide_obj")
            )
            
            if exp_config_data.get("enable_guardrails", False) and exp_config_data.get("enable_response_guardrails", False):
                inferencer = GuardRailsInferencer(inferencer, base_guardrails)

            retriever = Retriever(json_reader, embedding, vector_storage, inferencer, reranker)
            hierarchical = exp_config_data.get("chunking_strategy") == 'hierarchical'
            result = retriever.retrieve(
                gt_data_path, 
                "What is the patient's name?",
                int(exp_config_data.get("knn_num")), 
                hierarchical
            )

            # metrics
            batch_items = []
            for item in result:
                metrics = create_metrics(
                    exp_config_data.get("execution_id"),
                    exp_config_data.get("experiment_id"),
                    item.question,
                    item.gt_answer,
                    item.answer,
                    item.reference_contexts,
                    item.query_metadata,
                    item.answer_metadata,
                    item.guardrails_input_assessment,
                    item.guardrails_context_assessment,
                    item.guardrails_output_assessment,
                    exp_config_data.get("guardrail_id", ""),
                    item.guardrails_blocked
                )
                batch_items.append(metrics)
                if len(batch_items) >= 25:
                    write_batch_to_metrics_dynamodb(batch_items)
                    batch_items = []

            if len(batch_items) > 0:
                write_batch_to_metrics_dynamodb(batch_items)

            
            output = {"status": "success", "message": "Retriever completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during retriever process: {str(e)}")
            self.send_task_failure(str(e))



# metrics: to be removed later
def create_metrics(
    execution_id,
    experiment_id,
    question: str,
    gt_answer: str,
    answer: str,
    reference_contexts: List[str],
    query_metadata: Dict[str, int],
    answer_metadata: Dict[str, int],
    guardrail_input_assessment: Optional[Union[List[Dict], Dict]] = None,
    guardrail_context_assessment: Optional[Union[List[Dict], Dict]] = None,
    guardrail_output_assessment: Optional[Union[List[Dict], Dict]] = None,
    guardrail_id: Optional[str] = None,
    guardrail_blocked: Optional[str] = None
):
    metrics = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "execution_id": execution_id,
        "experiment_id": experiment_id,
        "question": question,
        "gt_answer": gt_answer,
        "answer": answer,
        "reference_contexts": reference_contexts,
        "query_metadata": query_metadata,
        "answer_metadata": answer_metadata,
        "guardrail_input_assessment": guardrail_input_assessment,
        "guardrail_context_assessment": guardrail_context_assessment,
        "guardrail_output_assessment": guardrail_output_assessment,
        "guardrail_id": guardrail_id,
        "guardrail_blocked": guardrail_blocked
    }
    return metrics


def write_batch_to_metrics_dynamodb(batch_items: List[Dict]) -> None:
    """Write a batch of items to DynamoDB."""
    logger.info(f"Writing batch of {len(batch_items)} items to DynamoDB")
    db = DynamoDB(config.get_experiment_question_metrics_table())
    db.bulk_write(batch_items)
