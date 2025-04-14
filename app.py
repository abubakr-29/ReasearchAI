from flask import Flask, render_template, request, jsonify
import os
import requests
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

# Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PORT = int(os.getenv("PORT", 10000))
MODEL_NAME = "sshleifer/distilbart-cnn-6-6"

# Initialize model lazily
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        print("Loading summarization model...")  # Changed from app.logger for Render visibility
        summarizer = pipeline(
            "summarization", 
            model=MODEL_NAME,
            device=-1  # Use CPU
        )
    return summarizer

def search_web(query):
    """Search the web using Tavily API"""
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": TAVILY_API_KEY},
            json={
                "query": query,
                "max_results": 3,
                "include_raw_content": True
            },
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        return [{
            "content": result.get("content", ""),
            "title": result.get("title", "Untitled"),
            "url": result.get("url", "#")
        } for result in data.get("results", [])]
    
    except Exception as e:
        print(f"Search error: {str(e)}")  # Changed from app.logger
        return []

def create_app():
    """Factory function to create Flask app for Gunicorn"""
    app = Flask(__name__)
    
    # Re-register routes in factory context
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
            summary = get_summarizer()(combined[:1000], max_length=150)[0]['summary_text']
            
            return jsonify({
                "answer": summary,
                "sources": [{"title": r["title"], "url": r["url"]} for r in results]
            })
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return jsonify({
                "answer": "❌ An error occurred while processing your request.",
                "sources": []
            })
    
    return app

# Create app instance
app = create_app()

# For development only
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT, threaded=True)
