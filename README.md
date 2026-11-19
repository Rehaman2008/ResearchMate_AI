# ResearchMate AI – Intelligent Research Paper Summarizer & Cloud AI Assistant

> 🎓 An AI-powered web application that helps students and researchers understand lengthy academic papers quickly, with an on-demand **☁️ Cloud AI Assistant** that can answer questions about the uploaded paper and possesses comprehensive knowledge across all topics.

---

## 📌 Main Features

1. **📄 Clean, Focused Research Paper Summarizer**:
   - Accepts academic research papers in PDF format.
   - Automatic text extraction using `pypdf`.
   - Generates an accurate, student-friendly 8-section breakdown (Title, Authors, Problem, Methodology, Dataset, Results, Limitations, 5 Key Takeaways).
   - "Not mentioned in the paper" strict grounding to prevent hallucinations.
   - One-click TXT download & copyable summary view.

2. **☁️ On-Demand Cloud AI Assistant**:
   - Tap the **☁️ Cloud Symbol** anytime to toggle the AI Assistant.
   - **Paper-Grounded Q&A**: When a PDF is uploaded, the Cloud AI answers questions directly based on the exact paper text.
   - **Universal Knowledge**: Possesses deep knowledge across **any topic** — Artificial Intelligence, Mathematics, Coding, Physics, History, Medicine, and Philosophy.
   - Quick prompt suggestion chips for fast queries.
   - Conversation history memory and clean clear options.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Web Framework | Streamlit |
| AI Engine | Google Gemini API (`google-genai` SDK) |
| Active Models | `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.8-flash` |
| PDF Extraction | `pypdf` |
| Environment | `python-dotenv` |

---

## 🚀 How to Run Locally

```bash
cd E:\doooo\ResearchMate_AI
.\.venv\Scripts\activate
streamlit run app.py
```
Open **http://localhost:8501** in your browser.
