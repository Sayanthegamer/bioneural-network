import os
import torch
from sentence_transformers import SentenceTransformer

class JournalPipeline:
    """
    Strict Unbatched Sequential Ingestion Pipeline.
    Embeds text via a frozen local CPU model (all-MiniLM-L6-v2) 
    and yields individual [1, 384] tensors to enforce the Batch Size = 1 constraint.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"[*] Initializing local feature extractor: {model_name}")
        # Loads the model locally. Will download (~80MB) on the very first run.
        self.encoder = SentenceTransformer(model_name)
        # Force CPU execution per architecture constraints
        self.encoder.to('cpu')

    def embed_sentence(self, text):
        """Converts a single string into a 384-dimensional tensor."""
        # Force the output to be a standard, writeable tensor
        with torch.no_grad():
            embedding = self.encoder.encode(text, convert_to_tensor=True)
            # .detach() ensures it's no longer tied to the inference graph
            # .clone() creates a fresh copy that PyTorch's autograd can manipulate
            return embedding.detach().clone().unsqueeze(0)

    def embed_tokens(self, text):
        """Converts a single string into a sequence of token embeddings [1, Seq_Len, 384]."""
        with torch.no_grad():
            features = self.encoder.tokenize([text])
            features = {k: (v.to('cpu') if isinstance(v, torch.Tensor) else v) for k, v in features.items()}
            outputs = self.encoder[0](features)
            token_embeddings = outputs['token_embeddings']
            return token_embeddings.detach().clone()

    def stream_journal(self, filepath):
        """Generator that yields (text, tensor) one line at a time."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Journal file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip blank lines and structural comments
                if not line or line.startswith('#'):
                    continue
                
                # Yield the raw text and its embedded tensor
                yield line, self.embed_sentence(line)


# ==========================================
# Pipeline Micro-Validation Routine
# ==========================================
if __name__ == "__main__":
    print("--- MNC Feature Pipeline Micro-Validation ---")
    
    pipeline = JournalPipeline()
    journal_path = os.path.join("data", "journal.txt")
    
    print("\nStarting Sequential Stream:")
    
    tensor_shapes_correct = True
    processed_count = 0
    
    # Iterate through the generator
    for text, tensor in pipeline.stream_journal(journal_path):
        processed_count += 1
        shape = list(tensor.shape)
        print(f"Ingested: '{text[:30]}...' -> Tensor Shape: {shape}")
        
        # Verify the Batch Size = 1 constraint and embedding dimension
        if shape != [1, 384]:
            tensor_shapes_correct = False
            
    if tensor_shapes_correct and processed_count > 0:
        print(f"\n[SUCCESS] Pipeline streamed {processed_count} lines. All tensors conform to [1, 384] unbatched requirement.")
    else:
        print("\n[FAIL] Pipeline streaming failed or structural constraints violated.")