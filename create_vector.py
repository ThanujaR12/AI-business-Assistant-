from openai import OpenAI
from get_apikey import get_api_key, is_valid_api_key
import os
import json

# Initialize the OpenAI client
api_key = get_api_key()
if not is_valid_api_key(api_key):
    raise ValueError("Invalid API key format")
client = OpenAI(api_key=api_key)

def chunk_text(text, chunk_size=4000):
    """Split text into chunks of approximately equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_vector_store(file_path):
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Split content into chunks
        chunks = chunk_text(content)
        
        # Create embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk,
                    encoding_format="float"
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error creating embedding for chunk: {str(e)}")
                continue
        
        if not embeddings:
            raise ValueError("No embeddings were created successfully")
        
        # Store the embeddings with the file name
        embedding_data = {
            'file': os.path.basename(file_path),
            'embeddings': embeddings
        }
        
        # Save embeddings to a file
        output_file = os.path.join('embeddings', f"{os.path.basename(file_path)}.json")
        os.makedirs('embeddings', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(embedding_data, f)
        
        print(f"Embeddings saved to {output_file}")
        return embedding_data
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        # Test with a sample file
        test_file = "Companyfiles/test.md"
        if os.path.exists(test_file):
            embedding = create_vector_store(test_file)
            print(f"Successfully created embedding for {test_file}")
        else:
            print(f"Test file {test_file} not found")
    except Exception as e:
        print(f"Error: {str(e)}")