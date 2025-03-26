from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.genai import types
import os
from tenacity import retry, wait_exponential, stop_after_attempt

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "http://localhost:3000"}})

api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyB1bWIahQuwnNbTMQOygJVPjx4Tu3WFyy8")
genai.configure(api_key=api_key)

client = genai.Client(api_key=api_key)

@app.route('/generate', methods=['POST'])
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(3))
def generate_content():
    try:
        data = request.json
        prompt = data.get('prompt', '')

        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            max_output_tokens=4096,
            response_mime_type="text/plain"
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        response = client.models.generate_content_stream(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=contents,
            config=generate_content_config
        )

        full_response = []
        for chunk in response:
            if chunk.text:
                full_response.append(chunk.text)

        if not full_response:
            return jsonify({'error': 'No content generated'}), 400

        return jsonify({'content': ''.join(full_response)}), 200

    except Exception as e:
        return jsonify({
            'error': f'Generation failed: {str(e)}',
            'advice': 'Please try again with a different prompt or wait a moment'
        }), 429

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
