from flask import Flask, request, render_template, jsonify, session
from flask_socketio import SocketIO
from flask_cors import CORS
import pandas as pd
import os
import json
from datetime import datetime
import numpy as np
import plotly.express as px
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()

# Global variables
vector_store = None
current_df = None

def parse_markdown_table(file_path):
    """Parse markdown table into pandas DataFrame"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    lines = content.strip().split('\n')
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]
    data = []
    
    for line in lines[2:]:  # Skip header and separator lines
        if '|' in line:
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(row) == len(headers):
                data.append(row)
    
    df = pd.DataFrame(data, columns=headers)
    
    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    # Convert date columns
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    
    return df

def create_vector_store(df):
    """Create vector store from DataFrame"""
    global vector_store
    
    # Convert DataFrame to documents
    loader = DataFrameLoader(df)
    documents = loader.load()
    
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Create vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def generate_visualization(df, question, chart_type=None):
    """Generate appropriate visualization based on question"""
    question_lower = question.lower()
    
    try:
        fig = None
        
        # Determine columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        date_cols = [col for col in df.columns if isinstance(df[col].dtype, pd.DatetimeDtype)]
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Time series analysis
        if any(word in question_lower for word in ['trend', 'over time', 'timeline']):
            if date_cols and numeric_cols.any():
                date_col = date_cols[0]
                value_col = [col for col in numeric_cols if 'amount' in col.lower() or 'sales' in col.lower()][0]
                data = df.groupby(date_col)[value_col].sum().reset_index()
                fig = px.line(data, x=date_col, y=value_col, title=f'{value_col} Over Time')
        
        # Distribution analysis
        elif any(word in question_lower for word in ['distribution', 'spread']):
            if numeric_cols.any():
                value_col = numeric_cols[0]
                if chart_type == 'histogram':
                    fig = px.histogram(df, x=value_col, title=f'{value_col} Distribution')
                else:
                    fig = px.box(df, y=value_col, title=f'{value_col} Distribution')
        
        # Comparison analysis
        elif any(word in question_lower for word in ['compare', 'comparison']):
            if categorical_cols.any() and numeric_cols.any():
                cat_col = categorical_cols[0]
                value_col = numeric_cols[0]
                data = df.groupby(cat_col)[value_col].sum().reset_index()
                if chart_type == 'pie':
                    fig = px.pie(data, values=value_col, names=cat_col, title=f'{value_col} by {cat_col}')
                else:
                    fig = px.bar(data, x=cat_col, y=value_col, title=f'{value_col} by {cat_col}')
        
        # Correlation analysis
        elif any(word in question_lower for word in ['correlation', 'relationship']):
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                               title=f'Correlation: {numeric_cols[0]} vs {numeric_cols[1]}')
        
        # Default visualization
        if fig is None and numeric_cols.any():
            value_col = numeric_cols[0]
            fig = px.histogram(df, x=value_col, title=f'{value_col} Distribution')
        
        if fig:
            fig.update_layout(
                title_x=0.5,
                margin=dict(t=50, l=50, r=50, b=50),
                showlegend=True,
                template='plotly_white'
            )
            return fig.to_json()
        
        return None
    
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        return None

def query_data(question):
    """Query the vector store and generate response with visualization"""
    global vector_store, current_df
    
    if vector_store is None or current_df is None:
        return {"answer": "Please upload data first.", "visualization": None}
    
    try:
        # Search relevant documents
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Generate text response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant. Answer questions based on the provided data context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a clear and concise answer based on the data:"}
            ]
        )
        
        answer = response.choices[0].message.content
        
        # Generate visualization
        visualization = generate_visualization(current_df, question)
        
        return {
            "answer": answer,
            "visualization": json.loads(visualization) if visualization else None
        }
        
    except Exception as e:
        return {"answer": f"Error analyzing data: {str(e)}", "visualization": None}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.endswith('.md'):
            return jsonify({'error': 'Please upload a markdown file'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.md')
        file.save(filepath)
        
        # Parse and create vector store
        global current_df
        current_df = parse_markdown_table(filepath)
        
        if current_df.empty:
            return jsonify({'error': 'No data found in file'}), 400
        
        create_vector_store(current_df)
        
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'columns': current_df.columns.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('question')
def handle_question(data):
    try:
        question = data.get('question', '')
        if not question:
            return {'error': 'No question provided'}
        
        response = query_data(question)
        socketio.emit('response', response)
        
    except Exception as e:
        socketio.emit('error', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5005) 