from flask import Flask, render_template, request, jsonify
import os
import requests
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
port = os.getenv("PORT")
MODEL_NAME = "sshleifer/distilbart-cnn-6-6"  # Smaller model for faster loading

# Initialize model lazily
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model=MODEL_NAME)
    return summarizer

def search_web(query):
    """Search the web using Tavily API"""
    url = "https://api.tavily.com/search"
    headers = {"Authorization": TAVILY_API_KEY}
    payload = {"query": query, "max_results": 3}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return [{
            "content": result["content"],
            "title": result.get("title", "Untitled"),
            "url": result.get("url", "#")
        } for result in data.get("results", [])]
    
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Search error: {str(e)}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_input = request.form['question'].strip()
        if not user_input:
            return jsonify({"answer": "Please enter a question", "sources": []})
        
        results = search_web(user_input)
        if not results:
            return jsonify({
                "answer": "❌ No relevant results found for your query.",
                "sources": []
            })
        
        combined = " ".join([r["content"] for r in results])
        summary = get_summarizer()(combined[:1000])[0]['summary_text']
        
        return jsonify({
            "answer": summary,
            "sources": [{"title": r["title"], "url": r["url"]} for r in results]
        })
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "answer": "❌ An error occurred while processing your request.",
            "sources": []
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, threaded=True)