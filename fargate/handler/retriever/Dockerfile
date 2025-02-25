# Use the official AWS Lambda Python 3.9 runtime base image
FROM --platform=linux/amd64 python:3.9-slim

# Create and set the working directory inside the container
WORKDIR /var/task

COPY pyproject.toml .

# Install dependencies into the /var/task directory (where Lambda expects them)
RUN pip install .

# Copy the necessary files and directories
COPY chunking/ chunking/
COPY config/ config/
COPY embedding/ embedding/
COPY fargate/ fargate/
COPY indexing/ indexing/
COPY logger/ logger/
COPY reader/ reader/
COPY retriever/ retriever/
COPY storage/ storage/
COPY fargate/handler/retriever/fargate_retriever_handler.py .

# Set environment variables
ENV PYTHONPATH=/var/task
ENV PYTHONUNBUFFERED=1

CMD ["python", "fargate_retriever_handler.py"]