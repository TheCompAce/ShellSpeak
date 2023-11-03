from transformers import AutoTokenizer, AutoModel, logging
import faiss
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en')



logging.set_verbosity_error()

def tokenize(text):
    return tokenizer.encode(text, add_special_tokens=True, max_length=8192, truncation=True)

def detokenize(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)

def segment_text(text, window_size, overlap):
    # Tokenize the text
    tokens = tokenize(text)
    
    # Create segments with overlap
    segments = []
    start_index = 0
    while start_index < len(tokens):
        end_index = min(start_index + window_size, len(tokens))
        segment_tokens = tokens[start_index:end_index]
        segments.append(detokenize(segment_tokens))
        start_index += (window_size - overlap)
    
    return segments

def encode_segments(segments):
    # Batch encode the text segments to get embeddings
    return model(**tokenizer(segments, return_tensors='pt', padding=True, truncation=True)).last_hidden_state.mean(dim=1).detach().numpy()

def sliding_search(file_text, query, window_size=8192, overlap=100, top_k=5):
    # Segment the file text and encode the segments
    segments = segment_text(file_text, window_size, overlap)
    segment_embeddings = encode_segments(segments)
    
    # Encode the query
    query_embedding = model(**tokenizer(query, return_tensors='pt', padding=True, truncation=True)).last_hidden_state.mean(dim=1).detach().numpy()

    # Find the most relevant segments
    most_relevant_segment_indices = perform_search_logic(query_embedding, segment_embeddings, top_k=top_k)
    
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

def find_relevant_file_segments(history_text, file_data, window_size=8192, overlap=100, top_k=5):
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
