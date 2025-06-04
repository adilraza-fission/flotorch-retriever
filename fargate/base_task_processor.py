import json
from abc import ABC, abstractmethod
import sys

class BaseFargateTaskProcessor(ABC):
    """
    Abstract base class for Fargate task processors.
    """

    def __init__(self,  experiment_id: str, execution_id: str, config_data: dict):
        """
        Initializes the task processor with task token and input data.
        Args:
            task_token (str): The Step Functions task token.
            input_data (dict): The input data for the task.
        """
        self.experiment_id = experiment_id
        self.execution_id = execution_id
        self.config_data = config_data

    @abstractmethod
    def process(self):
        """
        Abstract method to be implemented by subclasses for processing tasks.
        """
        raise NotImplementedError("Subclasses must implement the process method.")

    def send_task_success(self, output: dict):
        """
        Sends task success signal to Step Functions.
        Args:
            output (dict): The output data to send to Step Functions.
        """
        print(json.dumps({"status": "success", "output": output}))
        sys.exit(0)

    def send_task_failure(self, error_message: str):
        """
        Sends task failure signal to Step Functions.
        Args:
            error_message (str): The error message to send to Step Functions.
        """
        print(json.dumps({"status": "error", "output": error_message}))
        sys.exit(1)