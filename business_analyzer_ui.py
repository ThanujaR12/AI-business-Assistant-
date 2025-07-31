from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import pandas as pd
import os
import json
from datetime import datetime
import logging
import sys
from werkzeug.utils import secure_filename
from openai import OpenAI
from get_apikey import get_api_key, is_valid_api_key
from create_assistant import create_business_assistant
from create_vector import create_vector_store
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s\n%(pathname)s:%(lineno)d',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'business_uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize OpenAI and global variables
api_key = get_api_key()
if not is_valid_api_key(api_key):
    raise ValueError("OpenAI API key not found or invalid")

client = OpenAI(api_key=api_key)
assistant = create_business_assistant()
current_data = None
vector_store = None

def process_markdown_table(content):
    """Process a markdown table into a pandas DataFrame.
    
    Args:
        content (str): The markdown table content
        
    Returns:
        pd.DataFrame: The processed table as a DataFrame
        
    Raises:
        ValueError: If the table is invalid or empty
    """
    try:
        # Split into lines and clean
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 3:  # Need header, separator, and at least one data row
            raise ValueError("Table must have at least 3 lines (header, separator, data)")
            
        # Process header row
        header_row = lines[0]
        headers = [col.strip() for col in header_row.strip('|').split('|')]
        headers = [h.strip() for h in headers if h.strip()]  # Remove empty headers
        
        if not headers:
            raise ValueError("No valid headers found in table")
            
        # Validate separator row
        separator_row = lines[1]
        if not all('-' in cell for cell in separator_row.strip('|').split('|')):
            raise ValueError("Invalid separator row (should contain only dashes and optional colons)")
            
        # Process data rows
        data = []
        for line in lines[2:]:  # Skip header and separator
            if not line.strip('| '):  # Skip empty lines
                continue
                
            # Split and clean cells
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            
            # Handle empty cells and NA values
            cells = [None if not cell.strip() or cell.strip().upper() in ('NA', 'N/A') else cell.strip() 
                    for cell in cells]
                    
            # Ensure correct number of columns
            if len(cells) != len(headers):
                logger.warning(f"Row has {len(cells)} cells but expected {len(headers)}. Row content: {line}")
                # Pad or truncate to match header length
                if len(cells) < len(headers):
                    cells.extend([None] * (len(headers) - len(cells)))
                else:
                    cells = cells[:len(headers)]
                    
            data.append(cells)
            
        if not data:
            raise ValueError("No valid data rows found in table")
            
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Convert numeric columns
        for col in df.columns:
            # Try to convert to numeric, handling currency symbols
            try:
                # Remove currency symbols and commas
                cleaned = df[col].str.replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(cleaned, errors='ignore')
            except:
                pass  # Keep as string if conversion fails
                
        # Try to convert date columns
        date_patterns = ['date', 'time', 'day', 'month', 'year']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass  # Keep as string if conversion fails
                    
        return df
        
    except Exception as e:
        logger.error(f"Error processing markdown table: {str(e)}")
        logger.error(f"Table content preview: {content[:200]}...")
        raise ValueError(f"Failed to process markdown table: {str(e)}")

@app.route('/')
def home():
    return render_template('business_analyzer.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process the data."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Check file extension
        if not file.filename.lower().endswith(('.csv', '.md', '.txt')):
            return jsonify({'error': 'Invalid file type. Please upload a CSV or markdown file'}), 400
            
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            return jsonify({'error': 'File is empty'}), 400
            
        # Process based on file type
        if filename.lower().endswith('.csv'):
            try:
                df = pd.read_csv(filepath)
                if df.empty:
                    return jsonify({'error': 'CSV file contains no data'}), 400
            except pd.errors.EmptyDataError:
                return jsonify({'error': 'CSV file is empty or malformed'}), 400
            except Exception as e:
                return jsonify({'error': f'Error processing CSV file: {str(e)}'}), 400
        else:
            # For markdown files
            try:
                # Extract table content (everything between the first and last line containing '|')
                lines = content.split('\n')
                table_start = None
                table_end = None
                
                for i, line in enumerate(lines):
                    if '|' in line:
                        if table_start is None:
                            table_start = i
                        table_end = i
                
                if table_start is None or table_end is None:
                    return jsonify({'error': 'No table found in markdown file'}), 400
                    
                table_content = '\n'.join(lines[table_start:table_end + 1])
                df = process_markdown_table(table_content)
                
            except ValueError as ve:
                return jsonify({'error': str(ve)}), 400
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                return jsonify({'error': f'Error processing file: {str(e)}'}), 400
        
        # Store the DataFrame globally
        global current_data, vector_store
        current_data = df
        
        # Create vector store
        try:
            vector_store = create_vector_store(filepath)
            logger.info(f"Successfully processed file {filename} with {len(df)} rows and {len(df.columns)} columns")
            
            # Prepare sample data for preview
            sample_data = df.head().copy()
            
            # Convert datetime columns to string for JSON serialization
            for col in sample_data.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_data[col]):
                    sample_data[col] = sample_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            return jsonify({
                'message': 'File uploaded and processed successfully',
                'rows': len(df),
                'columns': df.columns.tolist(),
                'preview': sample_data.to_dict('records'),
                'data_loaded': True
            })
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return jsonify({'error': f'Error creating vector store: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

def get_conversational_response(question):
    """Generate friendly responses for conversational queries while maintaining business context."""
    question = question.lower().strip()
    
    # Greeting patterns
    if any(word in question for word in ['hello', 'hi', 'hey']):
        return """Hello! 👋 I'm your friendly business assistant. I can help analyze your data, create insights, or just chat! What would you like to do?"""
    
    # Gratitude patterns
    elif any(word in question for word in ['thanks', 'thank you']):
        return """You're welcome! 😊 I enjoy helping out, whether it's with business analysis or just a friendly chat. Feel free to ask me anything!"""
    
    # Farewell patterns
    elif any(word in question for word in ['bye', 'goodbye']):
        return """Goodbye! 👋 Thanks for chatting. Whether you need business insights or just want to talk, I'll be here when you return!"""
    
    # How are you patterns
    elif any(word in question for word in ['how are you', 'how\'re you', 'how do you do']):
        return """I'm doing great, thanks for asking! 😊 Ready to help with both business analysis and friendly conversation. How are you today?"""
    
    # Default casual response
    else:
        return """I'm here to help! We can analyze your business data, discuss insights, or just have a friendly chat. What's on your mind?"""

def generate_visualizations(df, question, analysis_text=""):
    """Generate relevant visualizations based on the question and analysis text."""
    visualizations = []
    try:
        # Common layout settings
        base_layout = {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'family': 'Inter', 'size': 12},
            'margin': {'t': 80, 'b': 80, 'l': 80, 'r': 80},
            'showlegend': True,
            'legend': {
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': 'rgba(0,0,0,0.1)',
                'borderwidth': 1
            },
            'hoverlabel': {'bgcolor': 'white'},
            'hovermode': 'closest'
        }

        # Combine question and analysis text for context
        combined_text = (question + " " + analysis_text).lower()

        # Identify key terms in the question
        trend_terms = ['trend', 'over time', 'timeline', 'growth', 'change', 'sales trend']
        distribution_terms = ['distribution', 'breakdown', 'proportion', 'percentage', 'share']
        comparison_terms = ['compare', 'comparison', 'versus', 'vs', 'difference']

        # First, try to identify datetime columns
        date_cols = []
        for col in df.columns:
            try:
                if df[col].dtype == 'datetime64[ns]':
                    date_cols.append(col)
                elif any(term in col.lower() for term in ['date', 'time', 'day', 'year']):
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
            except:
                continue

        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Find relevant metrics based on the question and column names
        metric_terms = {
            'sales': ['sales', 'revenue', 'amount', 'price', 'value', 'cart value', 'total'],
            'quantity': ['quantity', 'units', 'count', 'number', 'items'],
            'profit': ['profit', 'margin', 'earnings', 'income'],
            'cost': ['cost', 'expense', 'spending', 'price']
        }

        # Try to find the most relevant metric column
        relevant_metric = None
        metric_score = {}
        
        for col in numeric_cols:
            col_lower = col.lower()
            score = 0
            # Check if column name directly matches terms in the question
            if any(term in combined_text and term in col_lower for term in sum(metric_terms.values(), [])):
                score += 3
            # Check if column matches any metric category
            for category, terms in metric_terms.items():
                if any(term in col_lower for term in terms):
                    score += 2
                if any(term in combined_text for term in terms):
                    score += 1
            metric_score[col] = score

        if metric_score:
            relevant_metric = max(metric_score.items(), key=lambda x: x[1])[0]
        elif len(numeric_cols) > 0:
            relevant_metric = numeric_cols[0]

        # Generate visualization based on the question type and available data
        if any(term in combined_text for term in trend_terms) and date_cols:
            # Time series visualization
            date_col = date_cols[0]
            df_sorted = df.sort_values(date_col)
            
            if relevant_metric:
                # Group by date and calculate the metric
                grouped_data = df_sorted.groupby(date_col)[relevant_metric].sum().reset_index()
                
                visualizations.append({
                    'type': 'scatter',
                    'data': [{
                        'x': grouped_data[date_col].dt.strftime('%Y-%m-%d').tolist(),
                        'y': grouped_data[relevant_metric].tolist(),
                        'type': 'scatter',
                        'mode': 'lines+markers',
                        'name': relevant_metric.replace('_', ' ').title(),
                        'line': {'color': '#6C63FF', 'width': 2},
                        'marker': {'size': 8},
                        'hovertemplate': f'Date: %{{x}}<br>{relevant_metric.replace("_", " ").title()}: %{{y:,.2f}}<extra></extra>'
                    }],
                    'layout': {
                        **base_layout,
                        'title': {
                            'text': f'{relevant_metric.replace("_", " ").title()} Trend Over Time',
                            'font': {'size': 16, 'color': '#2C3E50'}
                        },
                        'xaxis': {
                            'title': 'Date',
                            'tickangle': -45,
                            'gridcolor': 'rgba(0,0,0,0.1)',
                            'type': 'category'
                        },
                        'yaxis': {
                            'title': relevant_metric.replace('_', ' ').title(),
                            'gridcolor': 'rgba(0,0,0,0.1)',
                            'showgrid': True
                        },
                        'height': 500
                    }
                })

        elif any(term in combined_text for term in distribution_terms) and len(categorical_cols) > 0:
            # Find most relevant categorical column
            category_col = None
            for col in categorical_cols:
                if any(term in col.lower() for term in ['category', 'type', 'product', 'department']):
                    category_col = col
                    break
            if not category_col:
                category_col = categorical_cols[0]

            if relevant_metric:
                grouped_data = df.groupby(category_col)[relevant_metric].sum().sort_values(ascending=False)
                total = grouped_data.sum()
                
                visualizations.append({
                    'type': 'pie',
                    'data': [{
                        'labels': grouped_data.index.tolist(),
                        'values': grouped_data.values.tolist(),
                        'type': 'pie',
                        'hole': 0.4,
                        'textinfo': 'label+percent',
                        'hoverinfo': 'label+value+percent',
                        'hovertemplate': '%{label}<br>Value: %{value:,.2f}<br>Percentage: %{percent:.1f}%<extra></extra>',
                        'marker': {
                            'colors': ['#4A90E2', '#6C63FF', '#2ECC71', '#F1C40F', '#E74C3C', 
                                     '#9B59B6', '#34495E', '#1ABC9C', '#E67E22', '#95A5A6']
                        }
                    }],
                    'layout': {
                        **base_layout,
                        'title': {
                            'text': f'Distribution of {relevant_metric.replace("_", " ").title()} by {category_col.replace("_", " ").title()}',
                            'font': {'size': 16, 'color': '#2C3E50'}
                        },
                        'height': 500,
                        'annotations': [{
                            'text': f'Total {relevant_metric.replace("_", " ").title()}: {total:,.2f}',
                            'showarrow': False,
                            'x': 0.5,
                            'y': -0.15,
                            'xref': 'paper',
                            'yref': 'paper',
                            'font': {'size': 12}
                        }]
                    }
                })

        else:
            # Default to bar chart for comparisons or general analysis
            if len(categorical_cols) > 0 and relevant_metric:
                # Find most relevant categorical column
                category_col = None
                for col in categorical_cols:
                    if any(term in col.lower() for term in ['category', 'type', 'product', 'department']):
                        category_col = col
                        break
                if not category_col:
                    category_col = categorical_cols[0]

                grouped_data = df.groupby(category_col)[relevant_metric].sum().sort_values(ascending=False)
                
                visualizations.append({
                    'type': 'bar',
                    'data': [{
                        'x': grouped_data.index.tolist(),
                        'y': grouped_data.values.tolist(),
                        'type': 'bar',
                        'name': relevant_metric.replace('_', ' ').title(),
                        'hovertemplate': f'{category_col.replace("_", " ").title()}: %{{x}}<br>{relevant_metric.replace("_", " ").title()}: %{{y:,.2f}}<extra></extra>',
                        'marker': {'color': '#6C63FF', 'opacity': 0.8}
                    }],
                    'layout': {
                        **base_layout,
                        'title': {
                            'text': f'{relevant_metric.replace("_", " ").title()} by {category_col.replace("_", " ").title()}',
                            'font': {'size': 16, 'color': '#2C3E50'}
                        },
                        'xaxis': {
                            'title': category_col.replace('_', ' ').title(),
                            'tickangle': -45,
                            'gridcolor': 'rgba(0,0,0,0.1)'
                        },
                        'yaxis': {
                            'title': relevant_metric.replace('_', ' ').title(),
                            'gridcolor': 'rgba(0,0,0,0.1)',
                            'showgrid': True
                        },
                        'height': 500
                    }
                })

        return visualizations

    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        return []

@socketio.on('analyze')
def handle_analysis(data):
    try:
        question = data.get('question', '').strip()
        
        # Handle casual conversation
        conversation_keywords = [
            'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye',
            'how are you', 'how\'re you', 'how do you do'
        ]
        
        if any(keyword in question.lower() for keyword in conversation_keywords):
            socketio.emit('analysis_response', {
                'answer': get_conversational_response(question),
                'status': 'complete'
            })
            return
            
        # For business analysis, check if data is loaded
        if current_data is None:
            socketio.emit('analysis_response', {
                'answer': """I'd love to help with your business analysis! But first, I'll need some data to work with. Could you please upload your business data file? In the meantime, we can chat about what kind of insights you're looking for! 😊""",
                'error': 'No data loaded'
            })
            return
        
        # Generate visualizations based on the question
        visualizations = generate_visualizations(current_data, question)
        
        # Prepare data context
        data_summary = f"Data columns: {', '.join(current_data.columns)}\n"
        data_summary += f"Number of rows: {len(current_data)}\n"
        
        content = f"""Analyze this business data and answer the following question:
{question}

Data Context:
{data_summary}

Sample data (first 5 rows):
{current_data.head().to_string()}

Please provide a clear, concise answer focusing on the key metrics and insights."""
        
        socketio.emit('analysis_status', {'status': 'Creating conversation thread...'})
        thread = client.beta.threads.create()
        
        socketio.emit('analysis_status', {'status': 'Sending question for analysis...'})
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content
        )
        
        socketio.emit('analysis_status', {'status': 'Processing analysis...'})
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Wait for completion
        import time
        start_time = time.time()
        max_wait_time = 60
        
        while (time.time() - start_time) < max_wait_time:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                for msg in messages:
                    if msg.role == "assistant":
                        # Process each content block
                        analysis_result = ""
                        for content_block in msg.content:
                            if hasattr(content_block, 'text'):
                                analysis_result += content_block.text.value + "\n"
                            elif hasattr(content_block, 'image_file'):
                                # Handle image file if needed
                                logger.info("Received image content in response")
                                continue
                        
                        if analysis_result.strip():
                            socketio.emit('analysis_response', {
                                'answer': analysis_result.strip(),
                                'status': 'complete',
                                'visualizations': visualizations
                            })
                            return
                        else:
                            socketio.emit('analysis_response', {
                                'answer': "I've analyzed the data but couldn't generate a text response. However, I've created some visualizations to help explain the insights.",
                                'status': 'complete',
                                'visualizations': visualizations
                            })
                            return
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                socketio.emit('analysis_response', {
                    'answer': """I ran into a small issue with the analysis. Would you like to try rephrasing your question? Or we can chat about what specific insights you're looking for! 😊""",
                    'error': f'Analysis failed with status: {run_status.status}'
                })
                return
            
            time.sleep(1)
            socketio.emit('analysis_status', {
                'status': f'Analysis in progress... ({int(time.time() - start_time)}s)'
            })
        
        socketio.emit('analysis_response', {
            'answer': """The analysis is taking longer than expected. Let's try a different approach! What specific aspect of your business would you like to focus on? 🤔""",
            'error': 'Analysis timed out'
        })
        
    except Exception as e:
        logger.error(f"Error in handle_analysis: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        socketio.emit('analysis_response', {
            'answer': """Oops! Something unexpected happened. But don't worry - we can try a different approach! Would you like to rephrase your question, or shall we chat about what you're trying to learn? 😊""",
            'error': f'An error occurred: {str(e)}'
        })

@socketio.on('get_data_summary')
def handle_data_summary():
    try:
        if current_data is None:
            socketio.emit('data_summary', {
                'error': 'No data loaded'
            })
            return
        
        # Create summary
        sample_data = current_data.head(5).copy()
        
        # Convert timestamps to strings
        for column in sample_data.columns:
            if sample_data[column].dtype == 'datetime64[ns]':
                sample_data[column] = sample_data[column].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        summary = {
            'rows': len(current_data),
            'columns': list(current_data.columns),
            'sample': sample_data.to_dict(orient='records')
        }
        
        socketio.emit('data_summary', {
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error in handle_data_summary: {str(e)}")
        socketio.emit('data_summary', {
            'error': f'Error creating summary: {str(e)}'
        })

if __name__ == '__main__':
    logger.info("="*50)
    logger.info(f"Starting Business Analyzer UI server at {datetime.now()}")
    logger.info(f"Server port: 5006")
    logger.info("="*50)
    socketio.run(app, debug=True, port=5006) 