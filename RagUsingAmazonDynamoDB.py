import boto3
import json
import numpy as np
import voyageai
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from decimal import Decimal
import PyPDF2
import re
import docx

# Constants
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.3
BEDROCK_MODEL_ID = "meta.llama3-8b-instruct-v1:0"  # or whichever model you're using

class VectorDB:
    def __init__(self, name, api_key=None):
        load_dotenv()
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.name = name
        self.query_cache = {}

        # Initialize DynamoDB client
        self.dynamodb = boto3.resource('dynamodb')
        
        # Sanitize the table name
        sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
        self.table_name = f"VectorDB_{sanitized_name}"[:255]
        
        self.table = self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        try:
            table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {'AttributeName': 'id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'id', 'AttributeType': 'S'}
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            table.meta.client.get_waiter('table_exists').wait(TableName=self.table_name)
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceInUseException':
                raise
            table = self.dynamodb.Table(self.table_name)
        return table

    def load_data(self, data):
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("Input data must be a list of dictionaries")
        
        chunked_data = []
        for item in data:
            if 'text' not in item or 'chunk_heading' not in item:    
                raise ValueError("Each item in data must have 'text' and 'chunk_heading' keys")
            chunked_data.extend(self._chunk_text(item['text'], item['chunk_heading']))
        
        self._embed_and_store(chunked_data)
        print("Vector database loaded and saved to DynamoDB.")

    def _chunk_text(self, text, heading, max_words=256):
        words = text.split()
        chunks = [words[i:i + max_words] for i in range(0, len(words), max_words)]
        return [{"chunk_heading": heading, "text": " ".join(chunk)} for chunk in chunks]

    def _embed_and_store(self, data):
        batch_size = 128
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Create texts for embedding
            texts = [item['text'] for item in batch]
            
            # Get embeddings
            embeddings = self.client.embed(texts, model="voyage-2").embeddings

            # Store in DynamoDB
            with self.table.batch_writer() as batch_writer:
                for j, (embedding, item) in enumerate(zip(embeddings, batch)):
                    batch_writer.put_item(
                        Item={
                            'id': f"{self.name}_{i+j}",
                            'embedding': self._float_to_decimal(embedding),
                            'metadata': json.dumps(item, ensure_ascii=False)
                        }
                    )

    def search(self, query, k=1, similarity_threshold=1):
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        all_items = self._get_all_items()
        if not all_items:
            raise ValueError("No data loaded in the vector database.")

        similarities = []
        for item in all_items:
            embedding = self._decimal_to_float(item['embedding'])
            similarity = np.dot(embedding, query_embedding)
            if similarity >= similarity_threshold:
                try:
                    metadata = json.loads(item['metadata'])
                    encoded_metadata = self._encode_unicode_dict(metadata)
                    similarities.append((similarity, encoded_metadata))
                except Exception as e:
                    print(f"Error processing item: {str(e)}")

        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Extract and filter relevant information
        relevant_info = self._extract_relevant_info(query, similarities[:k])
        
        return relevant_info

    def _extract_relevant_info(self, query, similarities):
        query_lower = query.lower()
        relevant_info = []

        # Define categories and their keywords
        categories = {
            'skills': ['skills', 'expertise', 'technologies', 'programming', 'languages'],
            'education': ['education', 'university', 'degree', 'bachelor'],
            'experience': ['experience', 'work', 'job', 'position', 'role'],
            'projects': ['project', 'achievement', 'accomplishment'],
            'personal': ['personal', 'hobbies', 'interests']
        }

        # Determine the relevant category based on the query
        relevant_category = next((cat for cat, keywords in categories.items() 
                                  if any(keyword in query_lower for keyword in keywords)), None)

        for similarity, metadata in similarities:
            text = metadata.get('text', '').lower()
            
            if relevant_category:
                # Extract information based on the relevant category
                if relevant_category == 'skills':
                    match = re.search(r'skills(.*?)(?:education|experience|languages|\Z)', text, re.DOTALL | re.IGNORECASE)
                elif relevant_category == 'education':
                    match = re.search(r'education(.*?)(?:skills|experience|\Z)', text, re.DOTALL | re.IGNORECASE)
                elif relevant_category == 'experience':
                    match = re.search(r'experience(.*?)(?:education|skills|\Z)', text, re.DOTALL | re.IGNORECASE)
                elif relevant_category == 'projects':
                    match = re.search(r'project[s]?(.*?)(?:education|skills|experience|\Z)', text, re.DOTALL | re.IGNORECASE)
                elif relevant_category == 'personal':
                    match = re.search(r'personal(.*?)(?:education|skills|experience|\Z)', text, re.DOTALL | re.IGNORECASE)
                
                if match:
                    relevant_text = match.group(1).strip()
                    relevant_info.append((similarity, relevant_text))
            else:
                # If no specific category is identified, return the most similar chunk
                relevant_info.append((similarity, text))

        return relevant_info
    
    def _get_all_items(self):
        items = []
        scan_kwargs = {}
        done = False
        start_key = None
        while not done:
            if start_key:
                scan_kwargs['ExclusiveStartKey'] = start_key
            response = self.table.scan(**scan_kwargs)
            items.extend(response.get('Items', []))
            start_key = response.get('LastEvaluatedKey', None)
            done = start_key is None
        return items

    def _encode_unicode_dict(self, d):
        return {self._encode_unicode(k): self._encode_unicode(v) for k, v in d.items()}

    def _encode_unicode(self, s):
        if isinstance(s, str):
            return s.encode('ascii', 'backslashreplace').decode('ascii')
        elif isinstance(s, dict):
            return self._encode_unicode_dict(s)
        elif isinstance(s, list):
            return [self._encode_unicode(item) for item in s]
        return s

    def _encode_char(self, char):
        if ord(char) < 128:
            return char
        else:
            return f"\\u{ord(char):04x}"

    @staticmethod
    def _float_to_decimal(float_list):
        return [Decimal(str(x)) for x in float_list]

    @staticmethod
    def _decimal_to_float(decimal_list):
        return [float(x) for x in decimal_list]

def extract_filename(file_path):
    # Find the position of the last '/' character
    last_slash_index = file_path.rfind('/')
    
    # Find the position of the '.pdf' extension
    pdf_extension_index = file_path.rfind('.pdf')
    
    # Extract the substring between the last '/' and '.pdf'
    # Add 1 to last_slash_index to exclude the '/' character
    filename = file_path[last_slash_index + 1 : pdf_extension_index]
    
    # Return the extracted filename
    return filename

class LlmFacade:
    def __init__(self, model_id):
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.model_id = model_id
        
        session = boto3.Session()
        region = session.region_name
        self.bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=region)
        print("Configured to use: AWS Bedrock Service")

    def invoke(self, prompt: str) -> str:
        return self.invoke_bedrock_llm(prompt)

    def invoke_bedrock_llm(self, prompt: str) -> str:
        body = json.dumps({
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": 0.9,
            "max_gen_len": self.max_tokens
        })
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            response_body = json.loads(response['body'].read())
            return response_body.get('generation', '')
        except ClientError as err:
            message = err.response['Error']['Message']
            print(f"A client error occurred: {message}")
            return "500: Request failed"

def answer_query_base(query, db, llm):
    documents, context = retrieve_base(query, db)
    prompt = f"""
        You are tasked with answering the following query based STRICTLY on the provided context. 
        Ignore any external knowledge or assumptions beyond the context:
        <query>
        {query}
        </query>
        <documents>
        {context}
        </documents>
        Ensure the answer uses only the above context. Avoid hallucinations.
        """
    return llm.invoke(prompt)
    
def retrieve_base(query, db, similarity_threshold=0.7):
    results = db.search(query, k=3, similarity_threshold=similarity_threshold)
    context = ""
    for similarity, metadata in results:
        if isinstance(metadata, dict) and 'text' in metadata:
            context += f"\n{metadata['text']}\n"
        elif isinstance(metadata, str):
            context += f"\n{metadata}\n"
        else:
            print(f"Unexpected metadata format: {metadata}")
    return results, context

def chatbot_response(prompt, db, llm):
    result = answer_query_base(prompt, db, llm)
    return result

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        vector_db = None  # Initialize vector_db to None
        content = None

        if filename.endswith('.txt'):
            content = extract_text_from_txt(os.path.join(folder_path, filename))
        elif filename.endswith('.docx'):
            content = extract_text_from_docx(os.path.join(folder_path, filename))
        elif filename.endswith('.pdf'):
            content = extract_text_from_pdf(os.path.join(folder_path, filename))
        else:
            print(f"Unsupported file type: {filename}")
            continue
        
        if content:
            try:
                vector_db = VectorDB(filename)
                vector_db.load_data([{'text': content, 'chunk_heading': filename}])
                print("Data successfully loaded into the vector database.")
            except Exception as e:
                print(f"An error occurred while loading data: {str(e)}")
                continue  # Skip to the next file if there's an error

        # Initialize LlmFacade
        llm = LlmFacade(BEDROCK_MODEL_ID)

        # Example queries
        queries = [
            "What is Ramakrishna's educational background?",
        ]

        # Only proceed with queries if vector_db was successfully created
        if vector_db:
            for query in queries:
                try:
                    response = chatbot_response(query, vector_db, llm)
                    print(response)
                except ValueError as e:
                    print(f"Error during search: {str(e)}")
        else:
            print(f"Skipping queries for {filename} as vector database could not be created.")

# def process_folder(folder_path):
#     for filename in os.listdir(folder_path):
#         # if filename.endswith(".txt"):  # You can add more file types if needed
#         #     file_path = os.path.join(folder_path, filename)
#         if filename.endswith('.txt'):
#             content = extract_text_from_txt(os.path.join(folder_path, filename))
#         elif filename.endswith('.docx'):
#             content = extract_text_from_docx(os.path.join(folder_path, filename))
#         elif filename.endswith('.pdf'):
#             content = extract_text_from_pdf(os.path.join(folder_path, filename))
#         else:
#             print(f"Unsupported file type: {filename}")
#             continue
        
#         try:
#             vector_db = VectorDB(filename)
#             vector_db.load_data([{'text': content, 'chunk_heading': filename}])
#             # vector_db.load_data(pdf_text)
#             print("Data successfully loaded into the vector database.")
#         except Exception as e:
#             print(f"An error occurred while loading data: {str(e)}")

#         # Initialize LlmFacade
#         llm = LlmFacade(BEDROCK_MODEL_ID)

#         # Example queries
#         queries = [
#             # "What are Ramakrishna's key skills and expertise?",
#             "What is Ramakrishna's educational background?",
#             # "What is Ramakrishna's work experience?",
#             # "What projects has Ramakrishna worked on?",
#             # "What are Ramakrishna's personal interests?"
#         ]
#         for query in queries:
#             try:
#                 response = chatbot_response(query, vector_db, llm)
#                 # print(f"\nQuery: {query}")
#                 # print("Response:")
#                 print(response)
#                 # print("-" * 50)
#             except ValueError as e:
#                 print(f"Error during search: {str(e)}")
                # # You might want to implement chunking for large files here
                # vector_db.add_item(
                #     id=filename,
                #     text=content,
                #     metadata={"filename": filename, "path": file_path}
                # )
                # print(f"Processed and added: {filename}")

# pdf_path = "data/RAMAKRISHNA_THADIVAKA.pdf"
# fileName = extract_filename(pdf_path)
#print(result)  # Output: RAMAKRISHNA_THADIVAKA

# vector_db = VectorDB(fileName)
# page_numbers = [1,2]

folder_path = r"C:\Users\Dell\OneDrive\Desktop\PythonApplications\RAG\data\FilesToExtract"
process_folder(folder_path)

# pdf_text = extract_text_from_pdf(pdf_path)
    # Load data into the vector database
# print(pdf_text)
# print(pdf_text.encode('utf-8', errors='replace'))



# Add this to check the contents of the database
# print("\nDisplaying all data in the vector database:")
# vector_db.display_data()




















# def search(self, query, k=RETRIEVAL_CHUNK_SIZE, similarity_threshold=SIMILARITY_THRESHOLD):
#         if query in self.query_cache:
#             query_embedding = self.query_cache[query]
#         else:
#             query_embedding = self.model.encode([query], convert_to_tensor=True)
#             self.query_cache[query] = query_embedding

#         all_items = self._get_all_items()
#         if not all_items:
#             raise ValueError("No data loaded in the vector database.")

#         similarities = []
#         for item in all_items:
#             embedding = self._decimal_to_float(item['embedding'])
#             similarity = util.cos_sim(query_embedding, np.array(embedding)).item()  # Use cosine similarity
#             if similarity >= similarity_threshold:
#                 try:
#                     metadata = json.loads(item['metadata'])
#                     encoded_metadata = self._encode_unicode_dict(metadata)
#                     similarities.append((similarity, encoded_metadata))
#                 except Exception as e:
#                     print(f"Error processing item: {str(e)}")

#         similarities.sort(reverse=True, key=lambda x: x[0])
        
#         # Extract and filter relevant information
#         relevant_info = self._extract_relevant_info(query, similarities[:k])
        
#         return relevant_info