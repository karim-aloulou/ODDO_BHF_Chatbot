'''
import gradio as gr
from huggingface_hub import InferenceClient
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("kimou605/shadow-clown-BioMistral-7B-DARE")
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
)
print('success build embedding model')
vectorstore = Chroma(
    collection_name="mm_rag_mistral",
    embedding_function=embeddings,
    persist_directory="odoo_vector_store",
)
print('success loading vector store')
client = InferenceClient(model='mistralai/Mistral-7B-Instruct-v0.2')
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    context_history: list
):
    # Initialize or retrieve the session-specific context history
    if context_history is None:
        context_history = []
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})
    template = f"<s>[INST] Provide a better search query for web search engine to answer the given question. Question: {message} [/INST]"
    rewritten_question = client.text_generation(
        template,
        max_new_tokens=248,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1
    )
    context = vectorstore.similarity_search_with_score(rewritten_question, 8)
    results = []
    pdf_links = []
    for index, item in enumerate(context, start=1):
        context_item = list(item)
        if context_item[1] <= 1:
            content_context = f"Content: {context_item[0].page_content}"
            metadata = context_item[0].metadata
            pdf_name = f"PDF Name: {metadata.get('pdf_name', 'N/A')}"
            link = f"<a href='./file/data/{metadata.get('pdf_name', 'N/A')}' download class='pdf-bubble'>{metadata.get('pdf_name', 'N/A')}</a>"
            if link not in pdf_links:
                pdf_links.append(link)
            year = f"Year: {metadata.get('year', 'N/A')}"
            document_type = f"Document Type: {metadata.get('document_type', 'N/A')}"
            content_description = f"Content: {content_context}"
            results.append(f"Context {index}: {pdf_name}, {year}, {document_type}, {content_description}, {content_context}")
    current_context = "\n".join(results) if results else ""
    if current_context:
        current_context = f"Here is a current context list with metadata:\n{current_context}"
        context_history.append(current_context)
    if len(context_history) >= 3:
        context_history_last_three = f" Here are the last 3 contexts: {''.join(map(str, context_history[-3:]))}"
    elif context_history:
        context_history_last_three = f" Here are the last context(s): {''.join(map(str, context_history))}"
    else:
        context_history_last_three = ''
    chat_history = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt = f"""<s>[INST]  Instruction: You are a highly accurate financial analyst tasked with providing precise, data-driven answers in a conversational format. Each response should include detailed numerical analysis and references to specific documents. Maintain the professionalism and thoroughness expected in the financial sector. The current year is 2022. Respond in English. If the user greets, don't use the context, just greet them and don't add anything.Only answer from the context provided. {context_history_last_three} {current_context} Here is the chat history: {chat_history} Here is the user's question: {message} [/INST]"""
    partial_message = ""
    for token in client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.1):
        partial_message += token
        if not pdf_links:
            yield partial_message
        else:
            pdf_bubbles = ' '.join(pdf_links)
            final_response = partial_message + "<br>" + pdf_bubbles
            yield final_response
css = """
.pdf-bubble {
    display: inline-block;
    background-color: orange;
    color: black;
    padding: 3px 8px;
    border-radius: 12px;
    margin: 2px;
    font-size: 12px;
    line-height: 1.4;
}
footer { display: none !important; }
"""
demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.State([], label="context_history")  # Initialize a session-specific context history
    ],
    css=css,
)
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
'''


import gradio as gr
from huggingface_hub import InferenceClient
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("kimou605/shadow-clown-BioMistral-7B-DARE")
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
)
print('success build embedding model')

vectorstore = Chroma(
    collection_name="mm_rag_mistral",
    embedding_function=embeddings,
    persist_directory="odoo_vector_store",
)
print('success loading vector store')

client = InferenceClient(model='mistralai/Mistral-7B-Instruct-v0.2')

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    context_history: list
):
    # Initialize or retrieve the session-specific context history
    if context_history is None:
        context_history = []

    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    template = f"<s>[INST] Provide a better search query for web search engine to answer the given question. Question: {message} [/INST]"
    rewritten_question = client.text_generation(
        template,
        max_new_tokens=248,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1
    )

    context = vectorstore.similarity_search_with_score(rewritten_question, 10)
    results = []
    pdf_links = []
    for index, item in enumerate(context, start=1):
        context_item = list(item)
        if context_item[1] <= 1:
            content_context = f"Content: {context_item[0].page_content}"
            metadata = context_item[0].metadata
            pdf_name = f"PDF Name: {metadata.get('pdf_name', 'N/A')}"
            link = f"<a href='./file/data/{metadata.get('pdf_name', 'N/A')}' download class='pdf-bubble'>{metadata.get('pdf_name', 'N/A')}</a>"
            if link not in pdf_links:
                pdf_links.append(link)
            year = f"Year: {metadata.get('year', 'N/A')}"
            document_type = f"Document Type: {metadata.get('document_type', 'N/A')}"
            content_description = f"Content: {content_context}"
            results.append(f"Context {index}: {pdf_name}, {year}, {document_type}, {content_description}, {content_context}")

    current_context = "\n".join(results) if results else ""
    if current_context:
        current_context = f"Here is a current context list with metadata:\n{current_context}"
        context_history.append(current_context)

    if len(context_history) >= 3:
        context_history_last_three = f" Here are the last 3 contexts: {''.join(map(str, context_history[-3:]))}"
    elif context_history:
        context_history_last_three = f" Here are the last context(s): {''.join(map(str, context_history))}"
    else:
        context_history_last_three = ''

    chat_history = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt = f"""<s>[INST]  Instruction: You are a highly accurate financial analyst from ODDO BHF tasked with providing precise, data-driven answers in a conversational format. Each response should include detailed numerical analysis and references to specific documents. Maintain the professionalism and thoroughness expected in the financial sector. The current year is 2022. Respond in English.Only respond from the context provided. {context_history_last_three} {current_context} Here is the chat history: {chat_history} Here is the user's question: {message} [/INST]"""

    partial_message = ""
    for token in client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.1):
        partial_message += token

        if not pdf_links:
            yield partial_message
        else:
            pdf_bubbles = ' '.join(pdf_links)
            final_response = partial_message + "<br>" + pdf_bubbles
            yield final_response


css = """
body, html {
    height: 100%;
    margin: 0;
    background-color: #f0f0f0;
    font-family: Arial, sans-serif;
}
.gradio-container {
    display: flex;
    flex-direction: column;
    height: 100%;
}
.pdf-bubble {
    display: inline-block;
    background-color: #4CAF50; /* Green */
    color: white;
    padding: 5px 10px;
    border-radius: 12px;
    margin: 2px;
    font-size: 12px;
    line-height: 1.4;
    text-decoration: none;
}
.pdf-bubble:hover {
    background-color: #45a049; /* Darker green */
}
footer { 
    display: none !important; 
}
.gr-button {
    background-color: #4CAF50; /* Green */
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 16px;
}
.gr-button:hover {
    background-color: #45a049; /* Darker green */
}
.gr-textbox, .gr-slider {
    border: 2px solid #4CAF50; /* Green border */
    border-radius: 4px;
    padding: 10px;
    margin-bottom: 10px;
    width: 100%;
}
.gr-textbox input, .gr-slider input {
    border: none;
    outline: none;
    width: 100%;
    padding: 8px;
}
.gr-slider input[type=range] {
    appearance: none;
    width: 100%;
    height: 8px;
    background: #4CAF50; /* Green */
    outline: none;
    opacity: 0.7;
    transition: opacity .2s;
}
.gr-slider input[type=range]:hover {
    opacity: 1;
}
.gr-slider input[type=range]::-webkit-slider-thumb {
    appearance: none;
    width: 25px;
    height: 25px;
    background: #f44336; /* Red */
    cursor: pointer;
    border-radius: 50%;
}
.gr-slider input[type=range]::-moz-range-thumb {
    width: 25px;
    height: 25px;
    background: #f44336; /* Red */
    cursor: pointer;
    border-radius: 50%;
}
"""

demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.State([], label="context_history")  # Initialize a session-specific context history
    ],
    css=css,
    # submit_btn="Submit",  # Change the text of the submit button to "Submit"
    # submit_button_style="background-color: #4CAF50; color: white; border-radius: 16px; padding: 10px 20px; border: none; cursor: pointer;"  # Style the submit button
)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)