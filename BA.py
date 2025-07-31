from openai import OpenAI
from get_apikey import get_api_key, is_valid_api_key
import pandas as pd
import json
import os
from create_assistant import create_business_assistant
from create_vector import create_vector_store
import logging

class BusinessAnalyzer:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.api_key = get_api_key()
        if not is_valid_api_key(self.api_key):
            raise ValueError("Invalid API key format")
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize assistant
        self.assistant = create_business_assistant()
        
        # Initialize storage
        self.data = None
        self.vector_store = None
        self.embeddings_path = 'embeddings'
        os.makedirs(self.embeddings_path, exist_ok=True)

    def load_data(self, file_path):
        """Load and process data from CSV or markdown file."""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:
                # Process markdown table
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Split into lines and clean
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                # Find header row
                header_row = None
                data_start = 0
                for i, line in enumerate(lines):
                    if '|' in line:
                        if header_row is None:
                            header_row = line
                            data_start = i + 2  # Skip separator row
                            break
                
                if not header_row:
                    raise ValueError("Could not find table headers")
                
                # Parse headers
                headers = [h.strip() for h in header_row.split('|')]
                headers = [h for h in headers if h]
                
                # Parse data rows
                data_lines = []
                for line in lines[data_start:]:
                    if '|' in line:
                        values = [v.strip() for v in line.split('|')]
                        values = [v for v in values if v]
                        if len(values) == len(headers):
                            processed_values = []
                            for v in values:
                                try:
                                    clean_v = v.replace('$', '').replace(',', '')
                                    if clean_v.strip():
                                        if '.' in clean_v:
                                            processed_values.append(float(clean_v))
                                        else:
                                            processed_values.append(int(clean_v))
                                    else:
                                        processed_values.append(None)
                                except ValueError:
                                    processed_values.append(v)
                            data_lines.append(dict(zip(headers, processed_values)))
                
                if not data_lines:
                    raise ValueError("No valid data rows found in table")
                
                # Convert to DataFrame
                self.data = pd.DataFrame(data_lines)
                
                # Try to convert date columns
                for col in self.data.columns:
                    if 'date' in col.lower():
                        try:
                            self.data[col] = pd.to_datetime(self.data[col])
                        except:
                            pass
            
            # Create vector store
            self.vector_store = create_vector_store(file_path)
            self.logger.info(f"Data loaded successfully from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False

    def analyze(self, query):
        """Analyze data based on user query."""
        try:
            if self.data is None:
                return "Please load data first using load_data(file_path)"

            print("\nStarting analysis...")
            
            # Prepare the data context
            print("Preparing data context...")
            data_summary = f"Data columns: {', '.join(self.data.columns)}\n"
            data_summary += f"Number of rows: {len(self.data)}\n"
            
            # Create message content with data context
            content = f"""Analyze this business data and answer the following question:
{query}

Data Context:
{data_summary}

Sample data (first 5 rows):
{self.data.head().to_string()}

Please provide a clear, concise answer focusing on the key metrics and insights."""

            # Create a thread for the conversation with timeout
            print("Creating conversation thread...")
            try:
                thread = self.client.beta.threads.create(timeout=10)
                print(f"Thread created: {thread.id}")
            except Exception as e:
                print(f"Error creating thread: {str(e)}")
                return f"Error creating thread: {str(e)}"

            # Add the user's message to the thread
            print("Sending message...")
            try:
                message = self.client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=content,
                    timeout=10
                )
                print("Message sent successfully")
            except Exception as e:
                print(f"Error sending message: {str(e)}")
                return f"Error sending message: {str(e)}"

            # Run the assistant with timeout
            print("Starting analysis run...")
            try:
                run = self.client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=self.assistant.id,
                    timeout=10
                )
                print(f"Analysis run started: {run.id}")
            except Exception as e:
                print(f"Error starting analysis: {str(e)}")
                return f"Error starting analysis: {str(e)}"

            # Wait for the run to complete with timeout
            print("\nWaiting for analysis to complete...")
            max_retries = 15
            retry_count = 0
            
            import time
            start_time = time.time()
            max_total_time = 60  # Maximum 60 seconds total

            while retry_count < max_retries and (time.time() - start_time) < max_total_time:
                try:
                    print(f"Checking status (attempt {retry_count + 1}/{max_retries})...")
                    run_status = self.client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id,
                        timeout=10
                    )
                    print(f"Current status: {run_status.status}")
                    
                    if run_status.status == 'completed':
                        print("Analysis completed successfully!")
                        break
                    elif run_status.status in ['failed', 'cancelled', 'expired']:
                        print(f"Analysis failed with status: {run_status.status}")
                        return f"Analysis failed with status: {run_status.status}"
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        print("Waiting 2 seconds before next check...")
                        time.sleep(2)
                except Exception as e:
                    print(f"Error checking status: {str(e)}")
                    return f"Error checking analysis status: {str(e)}"

            if retry_count >= max_retries:
                print("Analysis timed out after maximum retries")
                return "Analysis timed out. Please try again."
            
            if (time.time() - start_time) >= max_total_time:
                print("Analysis exceeded maximum total time")
                return "Analysis took too long. Please try again."

            # Get the assistant's response with timeout
            print("\nRetrieving analysis results...")
            try:
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread.id,
                    timeout=10
                )
                print("Retrieved messages successfully")
            except Exception as e:
                print(f"Error retrieving response: {str(e)}")
                return f"Error retrieving response: {str(e)}"

            # Process and return the latest assistant response
            print("Processing response...")
            for msg in messages:
                if msg.role == "assistant":
                    try:
                        response = msg.content[0].text.value
                        print("Response processed successfully")
                        print("\nAnalysis complete!")
                        return response
                    except (AttributeError, IndexError) as e:
                        print(f"Error processing response: {str(e)}")
                        return f"Error processing response: {str(e)}"

            print("No response found in messages")
            return "No response generated"

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            self.logger.error(f"Error in analyze: {str(e)}")
            return f"An error occurred: {str(e)}"

    def get_data_summary(self):
        """Get a summary of the loaded data."""
        if self.data is None:
            return "No data loaded"
        
        try:
            # Convert DataFrame to dictionary with date handling
            sample_data = self.data.head(5).copy()
            
            # Convert timestamps/datetime objects to strings
            for column in sample_data.columns:
                if sample_data[column].dtype == 'datetime64[ns]':
                    sample_data[column] = sample_data[column].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            summary = {
                "rows": len(self.data),
                "columns": list(self.data.columns),
                "sample": sample_data.to_dict(orient='records')
            }
            
            # Create a more readable string output instead of JSON
            output = "\nData Summary:\n"
            output += f"Total Rows: {summary['rows']}\n"
            output += f"\nColumns ({len(summary['columns'])}):\n"
            for col in summary['columns']:
                output += f"- {col}\n"
            
            output += "\nSample Data (first 5 rows):\n"
            output += sample_data.to_string()
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error creating data summary: {str(e)}")
            return f"Error creating data summary: {str(e)}"

def main():
    # Example usage
    analyzer = BusinessAnalyzer()
    
    # Command line interface
    while True:
        print("\nBusiness Analyzer Commands:")
        print("1. Load data file")
        print("2. Analyze data")
        print("3. Get data summary")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            file_path = input("Enter file path: ")
            if analyzer.load_data(file_path):
                print("Data loaded successfully!")
            else:
                print("Failed to load data.")
        
        elif choice == '2':
            if analyzer.data is None:
                print("Please load data first!")
                continue
            
            query = input("Enter your analysis query: ")
            result = analyzer.analyze(query)
            print("\nAnalysis Result:")
            print(result)
        
        elif choice == '3':
            summary = analyzer.get_data_summary()
            print("\nData Summary:")
            print(summary)
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 