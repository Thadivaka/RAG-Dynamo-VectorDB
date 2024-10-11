from dotenv import load_dotenv
import boto3
import anthropic
import numpy as np
import os

LLM_MAX_TOKENS = 2500
LLM_TEMPERATURE = 0.01
BEDROCK_MODEL_ID = 'anthropic.claude-3-haiku-20240307-v1:0'


class LlmFacade:
    def __init__(self, anthropic_api_key=None):
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        # Use Anthropic Claude via Anthropic Cloud if the key is set
        # if not, set up to use Anthropic Claude via Bedrock
        self.aws_bedrock = True

        if anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            self.aws_bedrock = False
            print("Configured to use: Anthropic Cloud Service")
        else:
            session = boto3.Session()
            region = session.region_name

            # Set the model id to Claude Haiku
            self.bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=region)
            print("Configured to use: AWS Bedrock Service")

    def invoke(self, prompt: str) -> str:
        if self.aws_bedrock == True:
            return self.invoke_aws_bedrock_llm(prompt)
        else:
            return self.invoke_anthropic_cloud_llm(prompt)

    def invoke_anthropic_cloud_llm(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.content[0].text

    def invoke_aws_bedrock_llm(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        inference_config = {
            "temperature": self.temperature,
            "maxTokens": self.max_tokens
        }
        converse_api_params = {
            "modelId": BEDROCK_MODEL_ID,
            "messages": messages,
            "inferenceConfig": inference_config
        }
        # Send the request to the Bedrock service to generate a response
        try:
            response = self.bedrock_client.converse(**converse_api_params)

            # Extract the generated text content from the response
            text_content = response['output']['message']['content'][0]['text']

            # Return the generated text content
            return text_content

        except ClientError as err:
            message = err.response['Error']['Message']
            print(f"A client error occured: {message}")
        return("500: Request failed")
    
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")  
llm = LlmFacade(anthropic_api_key=anthropic_api_key)
llm.invoke("how fast does a swallow fly")


def retrieve_base(query, db, similarity_threshold=0.7):
    results = db.search(query, k=3, similarity_threshold=similarity_threshold)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n{chunk['text']}\n"
    return results, context

def answer_query_base(query, db, llm):
    documents, context = retrieve_base(query, db)
    prompt = f"""
    You have been tasked with helping us to answer the following query: 
    <query>
    {query}
    </query>
    You have access to the following documents which are meant to provide context as you answer the query:
    <documents>
    {context}
    </documents>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already. 
    Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
    """
    return llm.invoke(prompt)

example_question = ["i have a billing question", "what capabilities are there", "who's cat is that"]
i = 0
results, context = retrieve_base(example_question[i], db)
print("Question:", example_question[i])
results