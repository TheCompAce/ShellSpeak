from datetime import datetime
import json
import os
import time
import spacy
import torch
from transformers import AutoTokenizer, AutoModel, logging, BertTokenizer, BertForSequenceClassification
import faiss
import numpy as np

from modules.save_load import save_settings

token_adjust = 2.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en').to(device)

from functools import partial
from multiprocessing import Pool, TimeoutError
# Load English tokenizer, POS tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")


logging.set_verbosity_error()

def tokenize(text):
    return tokenizer.encode(text, add_special_tokens=True, max_length=8192, truncation=True)

def detokenize(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)

def is_natural_break(token, next_token=None):
    """
    Determine if a token represents the end of a sentence or a natural break in the text.

    Args:
    - token (str): The current token to check.
    - next_token (str): The next token to check for cases where the end of sentence is followed by a quote or other character.

    Returns:
    - bool: True if the token is a natural break, False otherwise.
    """
    # Check for end of sentence punctuation followed by a space or a new line
    if token in {'.', '?', '!'}:
        if next_token in {'"', "'", '”', '’', ' ', '\n', None}:
            return True
    return False

def find_natural_break(tokens, start_index, window_size):
    """
    Adjust the window size to find a natural break in the text, considering punctuation and possible paragraph endings.

    Args:
    - tokens (list): The list of tokens.
    - start_index (int): The current start index in the tokens list.
    - window_size (int): The proposed window size.

    Returns:
    - int: The adjusted window size based on natural breaks.
    """
    # Begin at the end of the proposed window and move backwards
    for i in range(start_index + window_size, start_index, -1):
        # Check the token and the subsequent one to determine if there is a natural break
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        inputs = bert_tokenizer([tokens[i]], return_tensors="pt")
        current_token = bert_model(**inputs)
        inputs = bert_tokenizer([tokens[i + 1]], return_tensors="pt")
        next_token = bert_model(**inputs)
        # current_token = tokenizer.decode([tokens[i]]).strip()
        # next_token = tokenizer.decode([tokens[i + 1]]).strip() if i + 1 < len(tokens) else None
        if is_natural_break(current_token, next_token):
            # Return the new window size which ends at the natural break
            return i - start_index
    # If no natural break is found, return the original window size
    return window_size

def save_history_data(file_data, history_text, settings):
    use_indexing = settings.get('use_indexing', False)
    system_folder = 'system'
    data_file = os.path.join(system_folder, 'history.json')
    if use_indexing:
        # Check if the system folder exists, if not, create it
        if not os.path.exists(system_folder):
            os.makedirs(system_folder)

        entry = {
            'file_data': file_data,
            'history_text': history_text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        append_to_json(data_file, entry)

@staticmethod
def append_to_json(file_path, data):
    # Append data to a JSON file
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r+') as file:
                file_data = json.load(file)
                file_data['entries'].append(data)
                file.seek(0)
                json.dump(file_data, file, indent=4)
        else:
            with open(file_path, 'w') as file:
                json.dump({'entries': [data]}, file, indent=4)
    except Exception as e:
        print(f"Error saving data: {e}")


def load_faiss_index(index_file_path):
    if os.path.exists(index_file_path):
        return faiss.load_index_data(index_file_path)
    return None


#def tokenize(text):
    # Tokenize the text and return a dictionary of tensors
#    return tokenizer(text, add_special_tokens=True, max_length=8192, truncation=True, return_tensors="pt")

def load_index_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    vectorized_data = []
    for entry in data['entries']:
        # Encode each entry
        encoded_text = encode_text(entry['file_data'] + " " + entry['history_text'])
        # Append the output embeddings to vectorized_data
        vectorized_data.append(encoded_text.cpu().numpy())
    
    return np.vstack(vectorized_data)

def build_and_save_faiss_index(History_data, index_file_path):
    print("Starting index building process")
    elapsed_time = 0

    for h, History_item in enumerate(History_data["entries"]):
        start_time = time.time()  # Start the timer

        # Tokenize the text and move tensors to GPU if available
        tokenized_data = tokenizer(f"file_data: {History_item['file_data']}, history_text: {History_item['history_text']}", return_tensors='pt', padding=True, truncation=True).to(device)

        # Generate embeddings
        with torch.no_grad():
            model_output = model(**tokenized_data)
            embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()  # Move embeddings back to CPU

        # Check if embeddings is empty
        if embeddings.size == 0:
            print("No new data to add to the index.")
            continue

        # Load or create FAISS index
        if os.path.exists(index_file_path):
            index = faiss.read_index(index_file_path)
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])  # CPU index

        # Add embeddings to the FAISS index
        index.add(embeddings)

        # Save the index
        faiss.write_index(index, index_file_path)
        print("Data added to the index and index saved.")

        end_time = time.time()  # End the timer
        elapsed_time += end_time - start_time
        print(f"Time Left: {(end_time * (len(History_data['entries']) - h)):.2f} seconds")

    print(f"Index building process completed in {elapsed_time:.2f} seconds")

    return index


# Query FAISS Index
def query_index(index, query_vector, k=5):
    distances, indices = index.search(np.array([query_vector]), k)
    return indices

def preprocess_large_text(text, max_length=1000000):
    """
    If the text is larger than max_length, split the text into chunks using natural breaks and select a subset.

    Args:
    - text (str): The original text data.
    - max_length (int): The maximum length allowed for the text.

    Returns:
    - str: The processed text within the allowed length.
    """
    # If the text is already within the max length, return it as is
    if len(text) <= max_length:
        return text

    # Otherwise, process the text to reduce its size
    tokens = tokenize(text)
    chunks = []
    start_index = 0
    while start_index < len(tokens):
        # Use a window size (e.g., 10000 tokens) to find natural breaks for chunks
        window_size = min(10000, len(tokens) - start_index)
        end_index = find_natural_break(tokens, start_index, window_size) + start_index
        if end_index <= start_index:  # In case no natural break is found within the window
            end_index = start_index + window_size
        chunks.append(detokenize(tokens[start_index:end_index]))
        start_index = end_index

    # Now select a subset of chunks to get as close to the max length as possible
    selected_text = ""
    for chunk in chunks:
        if len(selected_text) + len(chunk) > max_length:
            break
        selected_text += chunk + " "

    return selected_text.strip()






def segment_text(text, base_window_size = 1000, base_overlap = 50):
    # Tokenize the text
    tokens = tokenize(text)
    
    # Create segments with dynamic overlap
    segments = []
    start_index = 0
    while start_index < len(tokens):
        window_size = find_natural_break(tokens, start_index, base_window_size)
        end_index = start_index + window_size
        segment_tokens = tokens[start_index:end_index]
        segments.append(detokenize(segment_tokens))
        # Dynamic overlap: adjust the next start index based on natural language understanding
        next_start_index = start_index + window_size
        next_window_size = find_natural_break(tokens, next_start_index, base_window_size)
        overlap = base_overlap if next_window_size == base_window_size else window_size - next_window_size
        start_index += (window_size - overlap)
    
    return segments

def encode_segments(segments):
    # Tokenize the segments and move tensors to GPU
    tokenized_inputs = tokenizer(segments, return_tensors='pt', padding=True, truncation=True).to(device)
    
    # Generate embeddings using the model
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.cpu().numpy()  # Move embeddings back to CPU
# def encode_text(text):
#    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
#    outputs = model(**inputs)
#    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def encode_text(text):
    # Tokenize the text, ensuring it returns a dictionary of tensors
    tokenized_inputs = encode_segments(segment_text(text))
    
    # Pass the tokenized inputs to the model and return the model's output
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        return outputs.last_hidden_state.mean(dim=1)


def sliding_search(file_text, query, settings, window_size=8192, overlap=100, top_k=5):
    # Segment the file text and encode the segments
    segments = segment_text(file_text, window_size, overlap)
    segment_embeddings = encode_segments(segments)
    
    # Encode the query
    query_embedding = model(**tokenizer(query, return_tensors='pt', padding=True, truncation=True)).last_hidden_state.mean(dim=1).detach().numpy()

    # Find the most relevant segments
    # most_relevant_segment_indices = perform_search_logic(query_embedding, segment_embeddings, top_k=top_k)
    most_relevant_segment_indices = find_relevant_file_segments(file_text, query, settings, window_size=window_size, overlap=overlap, top_k=top_k)
    
    # Retrieve the most relevant segments
    relevant_segments = [segments[i] for i in most_relevant_segment_indices]

    return relevant_segments

def perform_search_logic(query_embedding, segment_embeddings, top_k=5):
    # Create a FAISS index
    dimension = segment_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(segment_embeddings)  # Add the segment embeddings to the index

    # Search the index for the query embedding
    distances, indices = index.search(query_embedding, top_k)  # Search for the top k matches

    return indices.flatten().tolist()

def basic_search_algorithm(history_text, file_data, settings, window_size=8192, overlap=100, top_k=5):
    """
    Takes command history and file data to find and return the file segments that are most relevant.
    
    Args:
    - history_text (str): A string containing the concatenated command history.
    - file_data (str): The text of the file.
    - window_size (int): The size of each window for the sliding search.
    - overlap (int): The number of tokens to overlap between consecutive segments.
    - top_k (int): The number of top relevant segments to return.
    
    Returns:
    - list of str: A list of the most relevant file segments.
    """
    # Encode the command history
    history_embedding = model(**tokenizer(history_text, return_tensors='pt', padding=True, truncation=True)).last_hidden_state.mean(dim=1).detach().numpy()

    # Segment the file data and encode the segments
    file_segments = segment_text(file_data, window_size, overlap)
    file_segment_embeddings = encode_segments(file_segments)

    # Create a FAISS index for the file segment embeddings
    dimension = file_segment_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(file_segment_embeddings)  # Add the segment embeddings to the index

    # Search the index for the history embedding
    distances, indices = index.search(history_embedding, top_k)  # Search for the top k matches

    # Get the most relevant file segments based on the indices
    relevant_segments = [file_segments[i] for i in indices[0]]

    return relevant_segments

def find_relevant_file_segments(history_text, file_data, settings, window_size=8192, overlap=100, top_k=5):
    """
    Find relevant file segments based on history text and file data.

    Args:
    - history_text (str): Command history text.
    - file_data (str): Text of the file.
    - settings (dict): Application settings.
    - window_size (int): Size of each window for sliding search.
    - overlap (int): Overlap between windows.
    - top_k (int): Number of top relevant segments to return.

    Returns:
    - list of str: Most relevant file segments.
    """
    print(f"settings.get('use_indexing', False) = {settings.get('use_indexing', False)}")
    if settings.get('use_indexing', False):
        index_file_path = settings.get("index_file_path")
        if index_file_path and os.path.exists(index_file_path):
            faiss_index = load_faiss_index(index_file_path)
            if faiss_index:
                # Encode the history text
                query_vector = encode_segments([history_text])[0]  # As encode_segments expects a list of segments

                # Search the FAISS index
                distances, indices = faiss_index.search(query_vector, top_k)

                # Split the file data into segments
                file_segments = segment_text(file_data, window_size, overlap)

                # Retrieve the most relevant segments based on the FAISS search results
                relevant_segments = [file_segments[i] for i in indices.flatten()]

                # save_index_data(file_segments, history_text, settings)

                return relevant_segments

    # Fallback to basic search algorithm if FAISS index is not available
    return basic_search_algorithm(history_text, file_data, settings, window_size, overlap, top_k)


def needs_index_update(settings, json_file_path):
    # Implement logic to check if the index needs updating
    last_build_date = settings.get('last_build_date')
    
    # Get the latest timestamp from history.json
    latest_data_timestamp = get_latest_timestamp(json_file_path)
    
    if not last_build_date:
        # If the index has never been built, and there is data, return True
        return latest_data_timestamp > datetime.min
    else:
        # Check if there is new data since the last index build
        return datetime.strptime(last_build_date, '%Y-%m-%d %H:%M:%S') < latest_data_timestamp


def get_latest_timestamp(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            if data['entries']:
                latest_entry = max(data['entries'], key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'))
                return datetime.strptime(latest_entry['timestamp'], '%Y-%m-%d %H:%M:%S')
            else:
                # If there are no entries, return a default old date to trigger index building
                return datetime.min
    except Exception as e:
        print(f"Error reading the file {json_file_path}: {e}")
        # In case of error, return a default old date to trigger index building
        return datetime.min

# FAISS Index check and build prompt
def FAISS_Index_check_and_build(base_path, settings):
    if settings.get('use_indexing', False):
        system_folder_path = os.path.join(base_path, 'system')
        history_json_path = os.path.join(system_folder_path, 'history.json')
        settings["index_file_path"] = os.path.join(system_folder_path, 'faiss_index')

        if not os.path.exists(system_folder_path):
            os.makedirs(system_folder_path)

        if needs_index_update(settings, history_json_path):
            user_decision = input("A new FAISS index needs to be built. Do you want to build it now? (yes/no): ")
            if user_decision.lower() in ['yes', 'y']:
                print("Building FAISS index...")
                # Load history data
                history_data = load_index_data(history_json_path)

                # Filter data based on the last build date
                new_data = [entry for entry in history_data if datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') > last_build_datetime]
                                
                faiss_index = build_and_save_faiss_index(new_data, settings["index_file_path"])
                print("Index built and saved successfully.")
                settings['last_build_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_settings(settings, os.path.join(base_path, 'settings.json'))
                user_decision = input("Index built and saved successfully. Do you want to remove history? (yes/no): ")
                if user_decision.lower() in ['yes', 'y']:
                    user_decision = input("Are you sure, if you keep it, then it allows ypou to rebuild from scratch? (yes/no): ")
                    if user_decision.lower() in ['yes', 'y']:
                        os.remove(history_json_path)
            else:
                print("Skipping index building.")