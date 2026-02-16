# ScholarSync — AI-powered RAG over PDFs (LangChain + Groq + Chroma + Streamlit)

ScholarSync is a production-grade, low-latency Retrieval-Augmented Generation (RAG) system for context-aware Q&A over a folder of PDFs. It emphasizes **grounded answers**, **citations**, and **reduced hallucinations** via strict prompting + evidence packing.

---

## 1) Setup

### Prerequisites
- Python 3.10+
- A Groq API key

### Install
```bash
cd scholarsync
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
