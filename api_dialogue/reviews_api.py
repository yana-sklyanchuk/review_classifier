from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# Initialize OpenAI API
openai.api_key = 'YOUR_OPENAI_API_KEY'

@app.route('/feedback', methods=['POST'])
def feedback():
    user_input = request.json.get('input')
    
    # Process the user input and generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    
    reply = response['choices'][0]['message']['content']
    return jsonify({"response": reply})

if __name__ == '__main__':
    app.run(debug=True)