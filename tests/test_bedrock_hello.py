import json
import logging
from urllib import response
import boto3

from botocore.exceptions import ClientError

from src.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_embedding(model_id, body):
    logger.info(f"Generating embedding with Amazon Titan Text Embedding model: {model_id}")
    bedrock = boto3.client("bedrock-runtime")

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body = body,
        modelId = model_id,
        accept = accept,
        contentType = content_type
    )

    response_body = json.loads(response["body"].read())
    return response_body

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    model_id = "amazon.titan-embed-text-v2:0"
    input_text = "What are the different services that you offer?"
    
    body = json.dumps({
        "inputText": input_text,
        "embeddingTypes": ["binary"]
    })

    try:
        response = generate_embedding(model_id, body)
        print(f"Generated an embedding: {response['embeddingsByType']['binary']}") # returns binary embedding
        print(f"Input text: {input_text}")
        print(f"Input Token count:  {response['inputTextTokenCount']}")

    except ClientError as e:
        message = e.response['Error']['Message']
        logger.error(f"ClientError: {message}")
        print(f"ClientError: {message}")
    else:
        print("Embedding generation successful.")

if __name__ == "__main__":
    main()