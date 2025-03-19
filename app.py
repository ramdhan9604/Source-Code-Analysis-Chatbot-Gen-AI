from langchain.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load embeddings and initialize Chroma
embeddings = load_embedding()
persist_directory = "db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Initialize Groq and conversational chain
llm = ChatGroq(model="llama-3.3-70b-specdec", temperature=0.4)
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}), memory=memory)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():
    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")
        return jsonify({"response": str(user_input)})
    return jsonify({"response": "No input provided."})

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")
        return "Repository cleared."

    try:
        result = qa(input)
        print(result['answer'])
        return str(result["answer"])
    except Exception as e:
        return str(f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)