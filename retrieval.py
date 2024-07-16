from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from flask import Flask,request,render_template, jsonify
import json

app =  Flask(__name__)

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])

def query():
    try:
        
        data = request.get_json()
        query = data['message']
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        
        # query_vector = embeddings.embed_query(query)
        docs = new_db.similarity_search(query)
        
        
        results=[]
        for doc in docs:
            
            doc_info = {
                'Link': doc.metadata['Link'],
                'Description': doc.metadata['Description']
            }
            results.append(doc_info)
            
            
        print(results)
    

        
        
        return jsonify({'message': results})
        
    except Exception as e:
            print(f"Error during load: {e}")
    

if __name__ == "__main__":
    app.run(port=5000,debug=True)

