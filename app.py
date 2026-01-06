from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from chatbot_logic import get_response, check_reload_needed, load_and_train
import os

app = Flask(__name__)

# === Session Configuration ===
app.secret_key = "chatbot_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), "flask_session")  # persistent session folder
app.config["SESSION_PERMANENT"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = 3600  # 1 hour

Session(app)

# === Ensure sessions folder exists ===
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)

@app.route("/")
def index():
    if "chat_history" not in session:
        # store as list of (user_text, bot_text) tuples
        session["chat_history"] = []
        session.modified = True
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_reply():
    data = request.get_json()
    user_message = data.get("msg", "").strip()
    user_lang = data.get("lang", "english")

    if not user_message:
        return jsonify({"response": "⚠️ Please enter a message."})

    # Auto-load training data if needed
    if check_reload_needed():
        load_and_train()

    # Load chat history from session (list of (user, bot) tuples)
    history = session.get("chat_history", [])

    # Pass the full history to get_response — get_response will append the new pair
    response = get_response(user_message, user_lang, history)

    # Save updated history back to session
    session["chat_history"] = history
    session.modified = True  # Force Flask to save the updated session

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=False)