# 📚 Mistral RAG Chatbot

A robust, interactive web application built with Streamlit and LangChain that allows you to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

## ✨ Features
- **Upload PDF Books:** Seamlessly upload your PDF files for analysis.
- **Fast Mistral Small Model:** Utilizes Mistral AI's powerful and fast small model for generating accurate answers.
- **Chroma Vector Database:** Efficiently stores and retrieves document embeddings for context-aware responses.
- **Ask Questions:** Interactive chat interface to query your documents.
- **AI Powered Answers:** Generates answers based *only* on the provided context from the PDF.

## 🛠️ Technology Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Framework:** [LangChain](https://www.langchain.com/)
- **LLM:** [Mistral AI](https://mistral.ai/) (`mistral-small-2506`)
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Store:** Chroma
- **PDF Processing:** PyPDFLoader & RecursiveCharacterTextSplitter

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- A Mistral AI API Key

### Installation

1. Clone the repository (or download the project files).
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - **Windows:** `.\.venv\Scripts\activate`
   - **macOS/Linux:** `source .venv/bin/activate`
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the root directory of the project and add your API keys:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

### Running the App

Start the Streamlit application by running:
```bash
streamlit run app2.py
```
*(Note: If you want to run the other app version, use `streamlit run app.py` instead).*

## 💡 Usage

1. Open the app in your browser (usually `http://localhost:8501`).
2. Use the sidebar to upload a PDF file.
3. Wait for the processing to complete (the document is chunked and stored in a local Chroma database).
4. Once processed, start asking questions in the chat interface! 
5. Type `0` in the chat to end the session and clear the chat history.

## 📂 Project Structure

- `app2.py` / `app.py`: Main Streamlit application files.
- `requirements.txt`: Python dependencies needed to run the project.
- `chroma_db/`: Directory where the Chroma vector database stores embeddings locally.
- `.env`: Environment variables file (not included in version control).
