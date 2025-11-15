
import os
import sys
import time
from pathlib import Path
from typing import List, Dict
import json

import mlx.core as mx
from tokenizers import Tokenizer
from tqdm import tqdm


################################################################

def expand_tensor(input_tensor: mx.array, target_len: int) -> mx.array:
    """Pad tensor to target length."""
    current_len = input_tensor.shape[1]
    if current_len >= target_len:
        return input_tensor
    pad_len = target_len - current_len
    # Pad on the right: [(0, 0), (0, pad_len)]
    padding = [(0, 0), (0, pad_len)]
    expanded_tensor = mx.pad(input_tensor, padding)
    return expanded_tensor

################################################################

def cutoff_long_text_for_embedding_generation(text, encoding, cutoff=8192):
    import tiktoken
    encoded_text = encoding.encode(text)[:cutoff]
    decoded_text = encoding.decode(encoded_text)
    return decoded_text

def split_long_text_for_embedding_generation(text, encoding, cutoff=8192):
    encoded_texts = split_list(encoding.encode(text), cutoff)
    decoded_texts = [encoding.decode(encoded_text) for encoded_text in encoded_texts]
    return decoded_texts[:4]

################################################################

def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]

def get_embedding(text, encoding, model="text-embedding-ada-002"):
    for _ in range(5):
        try:
            text = cutoff_long_text_for_embedding_generation(text, encoding, cutoff=8192)
            return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
        except:
            print("Error generating embedding! Attempting again...")
            time.sleep(30)

################################################################
    
#############################################


class OpenAI_Encoder:
    def __init__(self, embedding_model="text-embedding-ada-002", **kwargs):
        self.embedding_model = embedding_model
        self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
        self.encoder_batch_size = 256

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        
        queries = [cutoff_long_text_for_embedding_generation(query, self.encoding, cutoff=8192) for query in queries]
        total_encoded_queries = []
        for query_chunks in tqdm(split_list(queries, self.encoder_batch_size)):
            try:
                encoded_queries = openai.Embedding.create(input=query_chunks, model=self.embedding_model)
            except:
                time.sleep(30)
                encoded_queries = openai.Embedding.create(input=query_chunks, model=self.embedding_model)
            encoded_queries = [query_encoding['embedding'] for query_encoding in encoded_queries['data']]
            total_encoded_queries += encoded_queries
        return np.array(total_encoded_queries)

        
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        
        passages = [passage['title'] + " " + passage['text'] for passage in corpus]
        passages = [cutoff_long_text_for_embedding_generation(passage, self.encoding, cutoff=8192) for passage in passages]
        total_encoded_passages = []
        for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size)):
            try:
                encoded_passages = openai.Embedding.create(input=passage_chunks, model=self.embedding_model)
            except:
                time.sleep(30)
                encoded_passages = openai.Embedding.create(input=passage_chunks, model=self.embedding_model)
            encoded_passages = [passage_encoding['embedding'] for passage_encoding in encoded_passages['data']]
            total_encoded_passages += encoded_passages
        return np.array(total_encoded_passages)
        
#############################################

class Voyager_Encoder:
    def __init__(self, embedding_model="voyage-01", **kwargs):
        self.embedding_model = embedding_model
        #self.encoding = tiktoken.encoding_for_model("voyage-01")
        self.encoder_batch_size = 64

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        
        #queries = [cutoff_long_text_for_embedding_generation(query, self.encoding, cutoff=4096) for query in queries]
        total_encoded_queries = []
        for query_chunks in tqdm(split_list(queries, self.encoder_batch_size)):
            try:
                encoded_queries = get_embeddings(query_chunks, model=self.embedding_model, input_type="query")
            except:
                time.sleep(30)
                encoded_queries = get_embeddings(query_chunks, model=self.embedding_model, input_type="query")
            #encoded_queries = [query_encoding for query_encoding in encoded_queries]
            total_encoded_queries += encoded_queries
        return np.array(total_encoded_queries)

        
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        
        passages = [passage['title'] + " " + passage['text'] for passage in corpus]
        #passages = [cutoff_long_text_for_embedding_generation(passage, self.encoding, cutoff=8192) for passage in passages]
        total_encoded_passages = []
        for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size)):
            try:
                encoded_passages = get_embeddings(passage_chunks, model=self.embedding_model, input_type="document")
            except:
                time.sleep(30)
                encoded_passages = get_embeddings(passage_chunks, model=self.embedding_model, input_type="document")
            encoded_passages = [passage_encoding for passage_encoding in encoded_passages]
            total_encoded_passages += encoded_passages
        return np.array(total_encoded_passages)
        
#############################################

class Cohere_Encoder:
    def __init__(self, truncation, embedding_model="embed-english-v3.0",  **kwargs):
        self.embedding_model = embedding_model
        #self.encoding = tiktoken.encoding_for_model(embedding_model)
        self.encoder_batch_size = 64
        self.co = cohere.Client(os.environ["COHERE_API_KEY"])
        self.truncation = truncation

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        
        #queries = [cutoff_long_text_for_embedding_generation(query, self.encoding, cutoff=4096) for query in queries]
        total_encoded_queries = []
        for query_chunks in tqdm(split_list(queries, self.encoder_batch_size)):
            try:
                encoded_queries = self.co.embed(texts=query_chunks, model=self.embedding_model, input_type="search_query", truncate=self.truncation)
            except:
                time.sleep(30)
                encoded_queries = self.co.embed(texts=query_chunks, model=self.embedding_model, input_type="search_query", truncate=self.truncation)
            encoded_queries = encoded_queries.embeddings
            total_encoded_queries += encoded_queries
        return np.array(total_encoded_queries)

        
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        
        passages = [passage['title'] + " " + passage['text'] for passage in corpus]
        #passages = [cutoff_long_text_for_embedding_generation(passage, self.encoding, cutoff=8192) for passage in passages]
        total_encoded_passages = []
        for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size)):
            try:
                encoded_passages = self.co.embed(texts=passage_chunks, model=self.embedding_model, input_type="search_document", truncate=self.truncation)
            except:
                time.sleep(30)
                encoded_passages = self.co.embed(texts=passage_chunks, model=self.embedding_model, input_type="search_document", truncate=self.truncation)
            encoded_passages = encoded_passages.embeddings
            total_encoded_passages += encoded_passages
        return np.array(total_encoded_passages)
    
#############################################

class M2_BERT_Encoder:
    def __init__(self, checkpoint=None, cfg=None, **kwargs):

        self.cfg = cfg
        self.max_seq_len = cfg.get("max_seq_len", 512)
        self.evaluation_max_seq_len = cfg.get("evaluation_max_seq_len", 512)

        ######################################
        # Load model from checkpoint (.pt or .bin)
        ######################################

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        # Add src directory to path for imports
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from configuration_bert import BertConfig
        from bert_layers import BertModel

        # Create config (following original m2-bert pattern)
        model_config = cfg.get('model_config', {})

        # Load config from HuggingFace format if available
        config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config_dict.update(model_config)
            self.config = BertConfig(**config_dict)
        else:
            # Fall back to cfg only - use from_pretrained to get base config
            self.config = BertConfig.from_pretrained('bert-base-uncased', **model_config)

        # Update config with all model_config items (to add custom attributes not in __init__)
        for key, value in model_config.items():
            self.config.update({key: value})

        # Create model
        self.model = BertModel(self.config)

        # Load weights from PyTorch checkpoint
        print(f"Loading weights from: {checkpoint_path}")
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.pytorch_loader import load_pytorch_bin

        checkpoint = load_pytorch_bin(checkpoint_path)

        # Extract model weights from checkpoint structure
        # Composer format (from Mosaic training): {'state': {'model': {...}}}
        if 'state' in checkpoint and 'model' in checkpoint['state']:
            state_dict = checkpoint['state']['model']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Clean up keys and convert to list of (name, array) tuples
        # Original checkpoint was from BertForSequenceClassification which has .bert attribute
        # Keys look like: "model.bert.embeddings.LayerNorm.weight"
        # Our BertModel expects: "embeddings.LayerNorm.weight"

        # Prepare weights list (clean keys)
        weights_list = []
        expanded_pos_emb = None

        for key, value in state_dict.items():
            # Clean key
            clean_key = key.replace("module.", "")  # DataParallel wrapper
            clean_key = clean_key.replace("model.", "", 1)  # Composer wrapper
            clean_key = clean_key.replace("bert.", "", 1)  # BertForSequenceClassification wrapper

            # Handle position embeddings expansion
            if clean_key == 'embeddings.position_embeddings.weight' and self.config.expand_positional_embeddings:
                orig_len = value.shape[0]
                target_len = self.config.max_position_embeddings
                if target_len > orig_len:
                    print(f"Expanding position embeddings from {orig_len} to {target_len}")
                    num_repeats = (target_len + orig_len - 1) // orig_len
                    expanded_pos_emb = mx.tile(value, (num_repeats, 1))[:target_len]
                    # Skip adding to weights_list - we'll set it manually after
                    continue

            weights_list.append((clean_key, value))

        # Load into model using load_weights (strict=False to skip missing/extra keys)
        self.model.load_weights(weights_list, strict=False)

        # Manually set expanded position embeddings if needed
        if expanded_pos_emb is not None:
            self.model.embeddings.position_embeddings.weight = expanded_pos_emb
            # Also expand position_ids to match
            self.model.embeddings.position_ids = mx.expand_dims(mx.arange(self.config.max_position_embeddings), axis=0)
            print(f"Set expanded position embeddings: {expanded_pos_emb.shape}")

        print("-------------------------------------------------")
        print(f"Pretrained Checkpoint: {checkpoint}")
        print("-------------------------------------------------")

        ######################################
        # Load tokenizer (no transformers needed)
        ######################################

        # Get tokenizer name from config (e.g., "bert-base-uncased")
        tokenizer_name = cfg.get('tokenizer_name', 'bert-base-uncased')

        # Look for tokenizer.json locally first
        tokenizer_dir = checkpoint_path.parent
        tokenizer_path = tokenizer_dir / "tokenizer.json"

        if not tokenizer_path.exists():
            # Try parent directory
            tokenizer_path = tokenizer_dir.parent / "tokenizer.json"

        if not tokenizer_path.exists():
            # Download from HuggingFace Hub
            print(f"Downloading tokenizer from HuggingFace: {tokenizer_name}")
            from huggingface_hub import hf_hub_download
            tokenizer_path = hf_hub_download(
                repo_id=tokenizer_name,
                filename="tokenizer.json",
                cache_dir=checkpoint_path.parent
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        print("M2 BERT Encoder Loaded")
        print(f"Max Sequence Length for Model: {self.max_seq_len}")
        print(f"Max Sequence Length for Evaluation: {self.evaluation_max_seq_len}")
    
    # Write your own encoding query function (Returns: Query embeddings as MLX array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> mx.array:

        all_embeddings = []
        total_batches = len(queries) // batch_size
        if len(queries) % batch_size != 0:
            total_batches = mx.add(mx.array(total_batches, dtype=mx.int32), mx.array(1, dtype=mx.int32)).item()

        for queries_chunk in tqdm(split_list(queries, batch_size), total=total_batches):

            # Tokenize batch
            encodings = self.tokenizer.encode_batch(queries_chunk, add_special_tokens=True)

            input_ids_list = []
            attention_mask_list = []

            for enc in encodings:
                ids = enc.ids
                mask = enc.attention_mask

                # Pad or truncate to evaluation_max_seq_len
                if len(ids) < self.evaluation_max_seq_len:
                    pad_len = self.evaluation_max_seq_len - len(ids)
                    ids = ids + [0] * pad_len  # Pad with 0
                    mask = mask + [0] * pad_len
                else:
                    ids = ids[:self.evaluation_max_seq_len]
                    mask = mask[:self.evaluation_max_seq_len]

                input_ids_list.append(ids)
                attention_mask_list.append(mask)

            # Convert to MLX arrays
            input_ids = mx.array(input_ids_list, dtype=mx.int32)
            attention_mask = mx.array(attention_mask_list, dtype=mx.int32)

            # Forward pass
            encoded_text = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract pooled output (index 1 for BERT)
            embedding = encoded_text[1] if isinstance(encoded_text, tuple) else encoded_text

            # Evaluate and append
            mx.eval(embedding)
            all_embeddings.append(embedding)

        # Concatenate all batches
        encoded_queries = mx.concatenate(all_embeddings, axis=0)

        # Check for NaNs
        if mx.any(mx.isnan(encoded_queries)).item():
            print("NaNs in encoded_queries")
            raise ValueError()

        return encoded_queries
    
    # Write your own encoding corpus function (Returns: Document embeddings as MLX array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> mx.array:

        # Combine title and text
        if self.evaluation_max_seq_len >= 512:
            passages = [doc['title'] + " " + doc['text'] for doc in corpus]
        else:
            passages = [doc['text'] for doc in corpus]

        # Reuse encode_queries
        return self.encode_queries(passages, batch_size)
    
#################################################################
    
def generate_together_embeddings(text: str, model_api_string: str, api_key: str):
    url = "https://api.together.xyz/api/v1/embeddings"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    session = requests.Session()
    response = session.post(
        url,
        headers=headers,
        json={
            "input": text,
            "model": model_api_string
        }
    )
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    return response.json()['data'][0]['embedding']
    
class Together_Encoder:
    def __init__(self, cfg, api_key, together_model_name, **kwargs):
        self.cfg = cfg
        self.api_key = api_key
        self.together_model_name = together_model_name
        self.encoder_batch_size = 1

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        
        total_encoded_queries = []
        for query_chunks in tqdm(split_list(queries, self.encoder_batch_size), total = len(queries) // self.encoder_batch_size + (1 if len(queries) % self.encoder_batch_size != 0 else 0)):
            try:
                encoded_queries = generate_together_embeddings(text=query_chunks[0], model_api_string=self.together_model_name, api_key=self.api_key)
            except:
                time.sleep(30)
                encoded_queries = generate_together_embeddings(text=query_chunks[0], model_api_string=self.together_model_name, api_key=self.api_key)
            assert type(encoded_queries) == list
            assert len(encoded_queries) == 768
            # breakpoint()
            total_encoded_queries.append(encoded_queries)
        return np.array(total_encoded_queries)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        
        passages = [passage['title'] + " " + passage['text'] for passage in corpus]
        
        total_encoded_passages = []
        for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size), total = len(passages) // self.encoder_batch_size + (1 if len(passages) % self.encoder_batch_size != 0 else 0)):
            # breakpoint()
            try:
                encoded_passages = generate_together_embeddings(text=passage_chunks[0], model_api_string=self.together_model_name, api_key=self.api_key)
            except:
                time.sleep(30)
                encoded_passages = generate_together_embeddings(text=passage_chunks[0], model_api_string=self.together_model_name, api_key=self.api_key)
            assert type(encoded_passages) == list
            assert len(encoded_passages) == 768
            total_encoded_passages.append(encoded_passages)
        return np.array(total_encoded_passages)
