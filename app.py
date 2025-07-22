from flask import Flask, request, jsonify, render_template
from rag_chain import rag_answer

app = Flask(__name__)

# ✅ Serve the HTML UI
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ✅ POST endpoint for asking questions
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Please provide a question"}), 400

    try:
        answer = rag_answer(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
