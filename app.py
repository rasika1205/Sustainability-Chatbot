from flask import Flask, render_template, request, url_for
import os
import google.generativeai as genai
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
api_key="your key"
genai.configure(api_key=api_key)
# Initialize Flask app
app = Flask(__name__)

# --- Document & Chain Initialization ---

# 1. Load the PDF (make sure the file is at static)
all_docs = []
pdf_folder = os.path.join("static", "uploads")
# Iterate over all files in the folder
for filename in os.listdir(pdf_folder):
    # Check if the file is a PDF (case-insensitive)
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        loader = PDFPlumberLoader(pdf_path)
        # loader.load() returns a list of document objects; extend the list
        all_docs.extend(loader.load())

# 3. Create the vector store from documents
index_dir = "faiss_index"  # Initialize the embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.split_documents(all_docs)
vector = FAISS.from_documents(documents, embedder)
vector.save_local(index_dir)
print("Created and saved new vector store.")
retriever = vector.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

llm = genai.GenerativeModel(
    model_name="gemini-1.5-flash"
)
# 5. Define the prompt for QA
PROMPT_TEMPLATE = """
1. Use the following pieces of context to answer the question at the end.

Context: {context}

Question: {question}

Helpful Answer:"""

def get_response(question):
    # Retrieve top documents
    retrieved_docs = retriever.invoke(question)

    # Concatenate context text
    context_text = "\n\n".join(
        f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', '')}\nPage: {doc.metadata.get('page', '')}"
        for doc in retrieved_docs
    )

    # Create the final prompt
    final_prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)

    # Call Gemini
    response = llm.generate_content(final_prompt)
    answer_text = response.text.strip() if response.text else "I don't know"

    # Extract source URL if any
    pdf_url = None
    if retrieved_docs:
        doc = retrieved_docs[0]
        metadata = doc.metadata
        source_doc = metadata.get("source", "")
        page_num = metadata.get("page", 0)
        normalized_source = source_doc.replace("\\", "/")
        if normalized_source.lower().startswith("static/"):
            normalized_source = normalized_source[len("static/"):]
        pdf_url = url_for("static", filename=normalized_source)
        pdf_url = f"{pdf_url}#page={page_num + 1}"
        print("Returned URL:", pdf_url)

    return answer_text, pdf_url


# --- Flask Routes ---


# Landing page route
@app.route("/")
def landing():
    return render_template("landing.html")


# Chat page route
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        question = request.form.get("question", "")
        answer, pdf_url = get_response(question)
        return render_template(
            "chat.html", question=question, answer=answer, pdf_url=pdf_url
        )
    else:
        return render_template("chat.html", question="", answer="", pdf_url="")


if __name__ == "__main__":
    app.run(debug=True)