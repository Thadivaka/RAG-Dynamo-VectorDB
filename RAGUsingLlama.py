from dotenv import load_dotenv
import boto3
import json
import os
import anthropic
import pickle
import json
import numpy as np
import voyageai
import PyPDF2
import sys
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline

sys.stdout.reconfigure(encoding='utf-8')

LLM_MAX_TOKENS = 2500
LLM_TEMPERATURE = 0.01
BEDROCK_MODEL_ID = 'anthropic.claude-3-haiku-20240307-v1:0'

class VectorDB:
    def __init__(self, name, model_name="meta-llama/Llama-2-7b-chat-hf"):
        load_dotenv()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def load_data(self, data):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts = [f"Heading: {item['chunk_heading']}\n\n Chunk Text:{item['text']}" for item in data]
        self._embed_and_store(texts, data)
        self.save_db()
        print("Vector database loaded and saved.")

    # TODO Change this function to limit text size sent to embeddeding to a max of 256 words
    def _embed_and_store(self, texts, data):
        batch_size = 8  # Reduced batch size due to potential memory constraints
        self.embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            self.embeddings.extend(embeddings.tolist())
        self.metadata = data

    def search(self, query, k=5, similarity_threshold=0.75):
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            inputs = self.tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_examples = []
        
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                example = {
                    "metadata": self.metadata[idx],
                    "similarity": similarities[idx],
                }
                top_examples.append(example)
                
                if len(top_examples) >= k:
                    break
        return top_examples

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])

    def display_data(self):
        """Method to display the embeddings and metadata."""
        if not self.embeddings or not self.metadata:
            print("No data loaded in the vector database.")
            return
        
        # Display embeddings and metadata
        for i, (embedding, meta) in enumerate(zip(self.embeddings, self.metadata)):
            print(f"Entry {i + 1}:")
            print(f"Embedding: {embedding}")
            print(f"Metadata: {meta}")
            print("-" * 50)

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def chunk_pdf_text(text, chunk_size=1000):
    """Splits the extracted text into chunks."""
    words = text.split()
    return [{'chunk_heading': f"Chunk {i+1}", 'text': ' '.join(words[i:i+chunk_size])} for i in range(0, len(words), chunk_size)]

# Load the PDF
pdf_text = extract_text_from_pdf('data/RAMAKRISHNA_THADIVAKA.pdf')

# Chunk the text
abbreviated_docs = chunk_pdf_text(pdf_text)
# print("abbreviated_docs...............", abbreviated_docs)
# print("abbreviated_docs...............", abbreviated_docs.encode('utf-8', 'ignore').decode('utf-8'))
# for doc in abbreviated_docs:
#     print("Document:", str(doc).encode('utf-8', 'ignore').decode('utf-8'))
    
# Load the data into the vector database
db = VectorDB("ramakrishna_docs")
db.load_data(abbreviated_docs)
    # Load the Anthropic documentation segments into a dictionary
# with open('data/RAMAKRISHNA_THADIVAKA.pdf', 'r') as f:
#     anthropic_docs = json.load(f)    

# len(anthropic_docs)
# abbreviated_docs = anthropic_docs[:10]


#     # Initialize the VectorDB
# db = VectorDB("anthropic_docs")
# # Import the document segments into the vector database
# db.load_data(abbreviated_docs)
# # db.display_data()
# # print("Database.....", db)
# len(db.embeddings)

class LlmFacade:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        print(f"Configured to use: {model_name}")

    def invoke(self, prompt: str) -> str:
        print("Inside invoke......")
        return self.invoke_llama_api(prompt)

    def invoke_llama_api(self, prompt: str) -> str:
        response = self.generator(prompt, max_length=self.max_tokens, temperature=self.temperature)
        return response[0]['generated_text']


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

# load_dotenv()
# llama_api_key = os.getenv("LLAMA_API_KEY")  # Change to LLAMA_API_KEY
# llm = LlmFacade(llama_api_key=llama_api_key)

load_dotenv()
llama_model_name = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
db = VectorDB("ramakrishna_docs", model_name=llama_model_name)
llm = LlmFacade(model_name=llama_model_name)

# print(llm.invoke("how fast does a swallow fly"))
example_question = ["i have a billing question", "what capabilities are there", "who's cat is that"]

# results, context = retrieve_base("profile", db)
# print("Question:", "profile")
# print("retrieve_base", results)

# result = answer_query_base("What projects has Ramakrishna worked on?", db, llm)
# print("Question:", "What projects has Ramakrishna worked on?")
# print("answer_query_base", result)

# result = answer_query_base("What Education has Ramakrishna studied", db, llm)
# print("Question:", "What Education has Ramakrishna studied")
# print("answer_query_base", result)

result = answer_query_base("What profile Ramakrishna has", db, llm)
print("Question:", "What profile Ramakrishna has")
print("answer_query_base", result)

# result = answer_query_base("Get Areas of Expertise", db, llm)
# print("Question:", "Get Areas of Expertise")
# print("answer_query_base", result)

# results, context = retrieve_base("what capabilities are there", db)
# print("Question:", "what capabilities are there")
# print("retrieve_base", results)

# result = answer_query_base("what capabilities are there", db, llm)
# print("Question:", "what capabilities are there")
# print("answer_query_base", result)

# results, context = retrieve_base("who's cat is that", db)
# print("Question:", "who's cat is that")
# print("retrieve_base..........", results)

# result = answer_query_base("who's cat is that", db, llm)
# print("Question:", "who's cat is that")
# print("answer_query_base.......", result)

# results, context = retrieve_base("i have a billing question", db)
# print("Question:", "i have a billing question")
# print("retrieve_base..........", results)

# result = answer_query_base("i have a billing question", db, llm)
# print("Question:", "i have a billing question")
# print("answer_query_base.......", result)
# i = 0
# results, context = retrieve_base(example_question[i], db)
# print("Question:", example_question[i])
# results

# result = answer_query_base(example_question[i], db, llm)
# print("Question:", example_question[i])
# result
# print(result)

# i = 1
# results, context = retrieve_base(example_question[i], db, 0.7)
# # print("Question:", example_question[i])
# results

# i = 1
# result = answer_query_base(example_question[i], db, llm)
# print("Question:", example_question[i])
# result
# print(result)

# i = 2
# results, context = retrieve_base(example_question[i], db, 0.7)
# # print("Question:", example_question[i])
# results

# result = answer_query_base(example_question[i], db, llm)
# print("Question:", example_question[i])
# result
# print(result)