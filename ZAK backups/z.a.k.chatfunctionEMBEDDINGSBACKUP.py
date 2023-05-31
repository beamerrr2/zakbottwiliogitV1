import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
import pinecone
from twilio.rest import Client
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from omegaconf import OmegaConf
from flask import Flask
from flask import request
from openai.embeddings_utils import cosine_similarity, get_embedding
import logging
from dataclasses import dataclass, field
import numpy as np
import openai
import pandas as pd

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

@dataclass
class ChatbotConfig:
    documents_file: str = "\\Users\\itsbe\\OneDrive\\Desktop\\Z.A.K Mastermind\\LongtermChatExternalSources\\CustomKndb2.txt"
    embedding_model: str = "text-embedding-ada-002"
    top_k: int = 3
    thresh: float = 0.7
    max_chars: int = 3000
    unknown_prompt: str = "unknown"  # Add this line


    completion_kwargs: dict = field(
        default_factory=lambda: {
            "engine": "text-davinci-003",
            "max_tokens": 200,
            "temperature": None,
            "top_p": None,
            "frequency_penalty": 1,
            "presence_penalty": 1,\
            
        }
    )

def compare_input_to_document(self, query: str) -> float:
    """
    Compare the user input to the document and return the cosine similarity.
    """
    engine = self.cfg.embedding_model
    query_embedding = get_embedding(query, engine=engine)
    similarity = cosine_similarity(self.document_embedding, query_embedding)
    return similarity

def read_documents(self, filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Assuming each line in the file represents a separate document
    documents = pd.DataFrame(lines, columns=["text"])

    # Calculate embeddings for each document
    engine = self.cfg.embedding_model
    documents["embedding"] = documents["text"].apply(lambda x: get_embedding(x, engine=engine))

    return documents

class Chatbot:
    openai.api_key = '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'
    documents_file: str = "\\Users\\itsbe\\OneDrive\\Desktop\\Z.A.K Mastermind\\LongtermChatExternalSources\\CustomKndb2.txt"
    embedding_model: str = "text-embedding-ada-002"
    top_k: int = 3
    thresh: float = 0.3
    max_chars: int = 3000
    unknown_prompt: str = "unknown"  # Add this line

    completion_kwargs: dict = field(
        default_factory=lambda: {
            "engine": "text-davinci-003",
            "max_tokens": 200,
            "temperature": None,
            "top_p": None,
            "frequency_penalty": 1,
            "presence_penalty": 1,
        }
    )
    def _init_documents(self):
        filepath = self.cfg.documents_file
        logger.info(f"loading embeddings from {filepath}...")
        self.documents = self.read_documents(filepath)
    logger.info(f"embeddings loaded.")

    
    def _init_unk_embedding(self):
        logger.info("Generating UNK token...")
        unknown_prompt = self.cfg.unknown_prompt
        engine = self.cfg.embedding_model
        self.unk_embedding = get_embedding(
            unknown_prompt,
            engine=engine,
        )
    
    def __init__(self, cfg: ChatbotConfig):
        # TODO: right now, the cfg is being passed as an omegaconf, is this what we want?
        self.cfg = cfg
        self._init_documents()
        self._init_unk_embedding()
    def rank_documents(self, documents: pd.DataFrame, query: str) -> pd.DataFrame:
        top_k = self.cfg.top_k
        thresh = self.cfg.thresh
        engine = self.cfg.embedding_model

        query_embedding = get_embedding(
            query,
            engine=engine,
        )
        documents["similarity"] = documents.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

        # sort the matched_documents by score
        matched_documents = documents.sort_values("similarity", ascending=False)

        # limit search to top_k matched_documents.
        top_k = len(matched_documents) if top_k == -1 else top_k
        matched_documents = matched_documents.head(top_k)

        # log matched_documents to the console
        logger.info(f"matched documents before thresh: {matched_documents}")

        # filter out matched_documents using a threshold
        if thresh:
            matched_documents = matched_documents[matched_documents.similarity > thresh]
            logger.info(f"matched documents after thresh: {matched_documents}")

        return matched_documents
    def read_documents(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # Assuming each line in the file represents a separate document
        documents = pd.DataFrame(lines, columns=["text"])

        # Calculate embeddings for each document
        engine = self.cfg.embedding_model
        documents["embedding"] = documents["text"].apply(lambda x: get_embedding(x, engine=engine))

        return documents
    
    
    # Other methods like rank_documents and read_documents should be here    

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def timestamp_to_datetime(unix_time):

    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''
    for i in short:
        if 'message' in i:
         output += '%s\n\n' % i['message']
    output = output.strip()
    return output

def similarity(v1, v2):
    return np.dot(v1, v2)/(norm(v1)*norm(v2))  

def fetch_memories(vector, logs, count, phone_number=None):
    scores = list()
    for i in logs:
        if phone_number and i['phone_number'] != phone_number:
            continue
        if vector == i['vector']:
            continue
        score = similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered

def summarize_memories(memories):  # summarize a block of memories into one payload
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    block = ''
    identifiers = list()
    timestamps = list()
    for mem in memories:
        block += mem['message'] + '\n\n'
        identifiers.append(mem['uuid'])
        timestamps.append(mem['time'])
    block = block.strip()
    prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', block)
    # TODO - do this in the background over time to handle huge amounts of memories
    notes = gpt3_completion(prompt)
    ####   SAVE NOTES
    vector = gpt3_embedding(block)
    info = {'notes': notes, 'uuids': identifiers, 'times': timestamps, 'uuid': str(uuid4()), 'vector': vector, 'time': time()}
    filename = 'notes_%s.json' % time()
    save_json('internal_notes/%s' % filename, info)
    return notes

def load_convo():
    files = os.listdir('nexus')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json('nexus/%s' % file)
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return ordered

def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def get_anticipation(flat):
    prompt = open_file('prompt_anticipate.txt').replace('<<INPUT>>', flat)
    return gpt3_completion(prompt)

def get_salience(flat):
    prompt = open_file('prompt_salience.txt').replace('<<INPUT>>', flat)
    return gpt3_completion(prompt)

def flatten_convo(conversation):
    convo = ''
    for i in conversation:
        if type(i) == dict and 'role' in i and 'content' in i:
            convo += '%s: %s\n' % (i['role'].upper(), i['content'])
    return convo.strip()

def gpt3_completion(prompt='Persona: Z.A.K. (Zero-wait Access to Knowledge) is a Sales and Customer service agent for DigitalMasterminds. Agenda: Convince business owners to utilize our agencys digital services. Task: Conduct cold outreach to potential leads, convert conversations to booking Zoom meetings, and encourage customers to inquire about our services via email. Z.A.K.s goal: Persuade customers to use our digital services. Take breaks to gauge interest and adjust approach.Keep communication concise and engaging.', engine="text-davinci-003", temp=0, top_p=1.0, tokens=800, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'Z.A.K.']):
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

def get_text_embedding(text):
    response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
    embedding = response['data'][0]['embedding']
    return embedding

if __name__ == "__main__":
    cfg = ChatbotConfig()
    chatbot_instance = Chatbot(cfg)

while True:
    openai.api_key = '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'
    conversation = []
    recent = get_last_messages(conversation, 4)
    a = input('\n\nUSER: ')
    conversation.append({'role': 'user', 'content': a})
    matched_documents = chatbot_instance.rank_documents(chatbot_instance.documents, a)
    print(f"\nMatched documents: {matched_documents}")
    flat = flatten_convo(conversation)
    anticipation = get_anticipation(flat)
    print('\n\nANTICIPATION: %s' % anticipation)
    salience = get_salience(flat)
    print('\n\nSALIENCE: %s' % salience)
    prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', recent).replace('<<ANTICIPATE>>',anticipation).replace('<<SALIENCE>>',salience)
    response = gpt3_completion(prompt)
    output = gpt3_completion(prompt)
    vector =  gpt3_embedding(output)
    conversation.append({'role': 'Z.A.K.', 'content': response})
    print('\nZ.A.K.: %s' % response) 
