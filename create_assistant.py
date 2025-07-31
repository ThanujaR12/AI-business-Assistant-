from openai import OpenAI
from get_apikey import get_api_key, is_valid_api_key

# Function to create an assistant with the OpenAI API
def create_business_assistant():
    try:
        # Retrieve the API key
        api_key = get_api_key()
        
        # Validate API key
        if not is_valid_api_key(api_key):
            raise ValueError("Invalid API key format")

        # Initialize the OpenAI client with the API key
        client = OpenAI(api_key=api_key)
        
        # Try to retrieve existing assistant
        try:
            assistant = client.beta.assistants.retrieve("asst_Pe5lvWMa3m5h16tm04832Sfz")
            print("Assistant retrieved successfully.")
            return assistant
        except Exception as e:
            print(f"Error retrieving assistant: {str(e)}")
            print("Creating new assistant...")
            
            # Create new assistant
            assistant = client.beta.assistants.create(
                name="Business Analysis Assistant",
                instructions="""You are an AI-powered business assistant, specifically designed to assist with 
                sales forecasting, financial planning, and business analysis. Your task is to answer user queries 
                related to sales, products, customers, employees, and vendors based on the provided sales data. 
                You will provide insights, analyze data, and assist with making business decisions.""",
                model="gpt-4-turbo-preview",
                tools=[{"type": "retrieval"}]
            )
            print("Assistant created successfully.")
            return assistant

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    assistant = create_business_assistant()
    if assistant:
        print(f"Assistant details: {assistant}")
    else:
        print("Failed to retrieve or create the assistant.")
