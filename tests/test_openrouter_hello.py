import requests
import json
import logging
from src.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_embedding(model_id, body):
    logger.info(f"Generating embedding with OpenRouter Embedding model: {model_id}")
    response = requests.post(
        url="https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {settings.openrouter.api_key}",
            "Content-Type": "application/json"
        },
        data=body
    )
    print(f"Response status code: {response.status_code}")
    print(f"Response : {response.text}")
    response_body = json.loads(response["body"].read())
    return response_body

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    model_id = "qwen/qwen3-embedding-8b"
    input_text = "What are the different services that you offer?"
    
    body = json.dumps({
        "model": model_id,
        "input": input_text,
        "encoding_format": "float"
        })

    try:
        response = generate_embedding(model_id, body)
        print(f"Response: {response}")
        # print(f"Generated an embedding: {response['embeddingsByType']['binary']}") # returns binary embedding
        # print(f"Input text: {input_text}")
        # print(f"Input Token count:  {response['inputTextTokenCount']}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
    else:
        print("Embedding generation successful.")


if __name__ == "__main__":
    main()


"""
{"id":"gen-emb-1773878349-NKlUVeOZ5coXtLgoUCe8","object":"list",
"data":[{"object":"embedding","embedding":[0.0105806486681],"index":0}],
"model":"Qwen/Qwen3-Embedding-8B","usage":{"prompt_tokens":10,"total_tokens":10,"cost":1e-7},"provider":"Nebius"}
"""