# Educational RAG Chatbot

A sophisticated Retrieval-Augmented Generation chatbot that helps primary school students learn about cells and chemistry of life using advanced hybrid search techniques.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL with PGVector extension:**
   ```bash
   # Install PostgreSQL and enable PGVector extension
   # Update connection string in chatbot.py if needed
   ```

3. **Set up environment:**
   ```bash
   cp .env.example .env
   # Add your OpenAI API key to .env file
   ```

4. **Run the web interface:**
   ```bash
   streamlit run app.py
   ```

5. **Or run the console version:**
   ```bash
   python chatbot.py
   ```

## What It Does

- Answers questions about cell biology using educational content
- Uses advanced hybrid retrieval combining semantic and keyword search
- Implements parent-child document strategy for better context
- Maintains conversation context with memory
- Provides age-appropriate responses for primary students
- Refuses to answer questions outside its knowledge scope

## Key Features

- **Hybrid Retrieval**: Combines vector similarity (60%) with BM25 keyword search (40%)
- **Parent-Child Documents**: Better context preservation while maintaining precision
- **Ensemble Search**: Advanced retrieval using multiple search strategies
- **Memory**: Remembers conversation history with session management
- **Safety**: Only answers based on provided educational content
- **Multilingual**: Supports responses in multiple languages
- **Web Interface**: Easy-to-use Streamlit interface

## Project Structure

```
SG/
├── app.py                  # Streamlit web interface
├── chatbot.py              # Main chatbot code
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── DEPLOYMENT.md          # Deployment guide
├── REQUIREMENTS_REPORT.md # Requirements compliance report
├── data/                  # Educational content (markdown)
├── ragas/                 # Evaluation framework
    ├── ragas_evaluation.py
    └── RAGAS_Evaluation_Report.md
```

## Evaluation

The chatbot has been tested with RAGAS metrics:
- Answer Relevancy: 86.9% ✅
- Context Precision: 84.0% ✅  
- Context Recall: 86.7% ✅
- Faithfulness: 75.0% (needs improvement)

Run evaluation: `cd ragas && python ragas_evaluation.py`

## Architecture

- **Vector Database**: PostgreSQL with PGVector extension
- **AI Model**: OpenAI GPT-4.1 with temperature=0 for consistent responses
- **Embeddings**: OpenAI text-embedding-3-small for document vectorization
- **Framework**: LangChain with advanced retrieval patterns
- **Retrieval Strategy**: 
  - Parent-Child document splitting (1024/2048 chunks)
  - Ensemble retrieval: Vector similarity (60%) + BM25 keyword (40%)
  - Context-aware query rewriting based on conversation history
- **Interface**: Streamlit web app + console interface
- **Storage**: LocalFileStore for document persistence

## Technical Implementation

### Advanced Retrieval System
- **Parent-Child Strategy**: Documents split into parent (2048 chars) and child (1024 chars) chunks
- **Hybrid Search**: Ensemble retriever combining:
  - Semantic similarity search (60% weight)
  - BM25 keyword search (40% weight)
- **Smart Query Processing**: Context-aware query rewriting for better retrieval
- **Session Management**: In-memory conversation history per session

### Educational Focus
- Age-appropriate language with emoji support
- Refuses out-of-scope questions with polite message
- Multilingual support (English, Mandarin, Malay)
- Factual grounding with source attribution

## Example Questions

- "What are cells and why are they important?"
- "Who discovered cells?"
- "How do microscopes work?"
- "What is protoplasm?"

## Usage Options

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

**Console Interface:**
```bash
python chatbot.py
```

**Evaluation:**
```bash
cd ragas && python ragas_evaluation.py
```

## Documentation

- `REQUIREMENTS_REPORT.md` - How we met all project requirements
- `ragas/RAGAS_Evaluation_Report.md` - Detailed evaluation results
- `DEPLOYMENT.md` - Simple deployment instructions