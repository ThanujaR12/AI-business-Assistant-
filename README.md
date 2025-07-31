# Business Analytics Assistant

An AI-powered business analytics platform that combines data analysis, visualization, and natural language processing to provide intelligent business insights.

## 🚀 Features

### Core Capabilities
- **AI-Powered Analysis**: OpenAI GPT-4 Turbo integration for intelligent data interpretation
- **Multi-Format Data Support**: Upload CSV, Excel, and Markdown table files
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Natural Language Queries**: Ask questions in plain English about your data
- **Real-time Processing**: Live data analysis with WebSocket support
- **Vector Database**: FAISS-powered semantic search for contextual understanding

### Business Intelligence
- **Sales Analytics**: Analyze product performance and pricing strategies
- **Financial Planning**: Generate insights for revenue optimization
- **Price Analysis**: Compare pricing data and identify trends
- **Data-Driven Decisions**: AI-powered recommendations based on business data

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework with SocketIO for real-time communication
- **Pandas & NumPy**: Data processing and analysis
- **OpenAI API**: GPT-4 Turbo for intelligent analysis
- **LangChain**: AI/ML framework integration
- **FAISS**: Vector database for semantic search

### Frontend
- **Angular 17**: Modern web framework for analytics dashboard
- **React**: Chat interface for conversational interactions
- **Plotly.js**: Interactive data visualizations
- **SocketIO**: Real-time client-server communication

### Data Processing
- **Multi-format Support**: CSV, Excel, Markdown tables
- **Automatic Data Cleaning**: Intelligent parsing and validation
- **Vector Embeddings**: Semantic understanding of data context

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API key

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/business-analytics-assistant.git
cd business-analytics-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config.ini.example config.ini
# Edit config.ini with your OpenAI API key
```

### Frontend Setup
```bash
# Install Angular CLI globally
npm install -g @angular/cli

# Navigate to Angular project
cd business-analytics-ui
npm install

# Start development server
ng serve
```

## ⚙️ Configuration

### API Keys
1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/)
2. Update `config.ini` with your API key:
```ini
[API]
apikey = your_openai_api_key_here
```

### Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key_here
FLASK_ENV=development
```

## 🚀 Usage

### Starting the Application

1. **Start the Flask backend**:
```bash
python business_analyzer_ui.py
```

2. **Start the Angular frontend**:
```bash
cd business-analytics-ui
ng serve
```

3. **Access the application**:
   - Analytics Dashboard: `http://localhost:4200`
   - Backend API: `http://localhost:5000`

### Using the Application

1. **Upload Data**: Drag and drop or select CSV, Excel, or Markdown files
2. **Ask Questions**: Use natural language to query your data
   - "What's the average price difference between final and initial prices?"
   - "Show me products with the highest discount percentage"
   - "Create a visualization of price trends"
3. **View Insights**: Get AI-generated analysis and interactive visualizations
4. **Export Results**: Download charts and analysis reports

## 📊 Sample Data

The application comes with sample Walmart product data including:
- Product pricing information
- Final vs initial price comparisons
- Discount analysis capabilities

## 📁 Project Structure

```
business-analytics-assistant/
├── business_analyzer_ui.py      # Main Flask application
├── BA.py                        # Business analyzer core logic
├── create_assistant.py          # OpenAI assistant setup
├── create_vector.py             # Vector database creation
├── business-analytics-ui/       # Angular frontend
├── business-chat-ui/           # React chat interface
├── CompanyFiles/               # Sample data files
├── templates/                  # HTML templates
└── requirements.txt            # Python dependencies
```

## 🔌 API Endpoints

### Data Upload
- `POST /upload` - Upload data files

### Analysis
- `POST /analyze` - Perform data analysis
- `GET /data-summary` - Get data overview

### WebSocket Events
- `analyze` - Real-time analysis requests
- `get_data_summary` - Data summary requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 Turbo API
- Plotly for interactive visualizations
- Angular and React communities
- Flask and Python ecosystem

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in the `/docs` folder
- Review the sample data and examples

## 🔮 Roadmap

- [ ] Enhanced visualization options
- [ ] Multi-language support
- [ ] Advanced AI models integration
- [ ] Mobile application
- [ ] Enterprise features
- [ ] API rate limiting and optimization

---

**Built with ❤️ for data-driven business decisions** 