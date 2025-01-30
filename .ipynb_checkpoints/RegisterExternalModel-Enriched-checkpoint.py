#
# Register ChatGPT endpoint
#
import os
import mlflow
import mlflow.pyfunc
import requests

import pandas as pd
import requests
import mlflow.pyfunc

# Define the endpoint details
endpoint_url = "https://api.openai.com/v1/chat/completions"
api_key = os.environ['OPENAI_API_KEY']  #"<your_openai_api_key>"


class ChatGPTModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, endpoint_url, api_key):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def predict(self, context, model_input):
        # Convert input to a list of strings (assumes DataFrame or Series input)
        if isinstance(model_input, pd.DataFrame):
            messages = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, pd.Series):
            messages = model_input.tolist()
        elif isinstance(model_input, list):
            messages = model_input
        else:
            raise ValueError("Unsupported input format for model_input")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": message} for message in messages],
        }
        response = requests.post(self.endpoint_url, headers=headers, json=payload)
        
        # Raise an error for bad HTTP responses
        response.raise_for_status()

        # Extract the API response
        return [choice["message"]["content"] for choice in response.json()["choices"]]


# Save the wrapped model to a local directory
model_path = "chatgpt_model3"
mlflow.pyfunc.save_model(
    path=model_path,
    python_model=ChatGPTModelWrapper(endpoint_url, api_key),
)

# Example metrics and tags

metrics = {
    "accuracy": 0.94,
    "bleu_score": 0.85,
    "rouge_l": 0.72,
    "perplexity": 15.2,
    "inference_time_per_token": 0.004,
    "win_rate_against_baseline": 0.78,
    "toxicity_rate": 0.02,
    "hallucination_rate": 0.08,
    "engagement_score": 4.7
}

parameters = {
    "model_size": "40B",
    "batch_size": 64,
    "learning_rate": 5e-5,
    "max_tokens": 1024,
    "training_data_size": "570GB",
    "optimizer": "AdamW",
    "dropout_rate": 0.1
}

tags = {
    "model_type": "ChatGPT",
    "developer": "Ahmet Gyger",
    "use_case": "Text Summarization",
    "version": "v3.0"
}

# Log and register the model
with mlflow.start_run() as run:
    registered_name=f"ChatGPT_Model3"
    
    # Log metrics
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
        
    # Log parameters
    for key, value in parameters.items():
        mlflow.log_param(key, value)

    
    mlflow.pyfunc.log_model(
        artifact_path=model_path,
        python_model=ChatGPTModelWrapper(endpoint_url, api_key),
        registered_model_name=registered_name,
        input_example=pd.DataFrame(["Hello, how are you?"], columns=["prompt"])
    )

    
    # Use MLflow Client to set registered model tags
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_tag(
        name=registered_name,
        key="use_case",
        value="ChatGPT automated predictions"
    )
    client.set_registered_model_tag(
        name=registered_name,
        key="priority",
        value="high"
    )
    
    
    # Get the latest model version
    model_version = client.get_latest_versions(name=registered_name, stages=["None"])[0].version

    # Set tags for the specific model version
    for key, value in tags.items():
        client.set_model_version_tag(
            name=registered_name,  
            version=model_version,
            key=key,  
            value=value  
        )

import mlflow.pyfunc
import pandas as pd

# Load the registered model
model = mlflow.pyfunc.load_model("models:/ChatGPT_Model3/latest")

# Test predictions
input_data = pd.DataFrame(["Hello, how are you?"], columns=["prompt"])
result = model.predict(input_data)
print(result)
