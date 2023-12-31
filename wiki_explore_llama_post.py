import requests
import re

import os
import sys
from transformers import AutoTokenizer
import transformers
import torch
import traceback

import chromadb #pip install...
import wikipedia

#You need a folder ./chroma_db to make this work
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="wiki")

model_path = "meta-llama/Llama-2-13b-chat-hf"
#load the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

current_search_term = "Biology"
filename = "biology.txt"
question_file = "biology_musings.txt"
num_results = 25 #change tihs to whatever you want. More means more options, fewer means more relevant
num_iterations = 200 #more iterations means more information, but higher chance of getting into WEIRD territory

prompt = "Given the following Wikipedia summary text, generate five wikipedia search terms related to the text that someone could use to explore the topic further. Return the search terms as a bulleted list without any explanation: "

prompt_question = "Given the following summary text, generate a question that you have about it. Return the question as a bullet point without any explanation: "
 
prompt_search = "Given the following question that was asked by a curious wikipedia explorer, generate three search terms for Wikipedia that might be used to answer the question. Return the search terms as a bulleted list without any explanation: "

prompt_chroma = "You will be given text and a question. Determine if a suitable answer to the question is in the text. If so, generate an answer using information from the text. If not, state your opinion but be very clear that you are stating an opinion. The text is: "

meta = [tokenizer,pipeline,prompt]

#Split a long string into chunks
def split_string_into_chunks(text, max_words_per_chunk):
    words = text.split()
    chunks = [words[i:i + max_words_per_chunk] for i in range(0, len(words), max_words_per_chunk)]
    return [' '.join(chunk) for chunk in chunks]

def generate_indexed_strings(name, num_strings):
    indexed_strings = [f"{name}{i}" for i in range(num_strings)]
    return indexed_strings

"""
    Adds text with page_title to the chroma DB
"""
def add_to_chroma(text,page_title,collection):
    splitScript = split_string_into_chunks(text,200)
    ids = generate_indexed_strings(page_title,len(splitScript))
    collection.add(documents=splitScript,ids=ids)

"""
    Queries the chroma DB and gets the top N results
"""
def get_top_N_from_chroma(collection, query, N):
    chroma_results = "".join(collection.query(query_texts=[query],n_results=N)['documents'][0])
    return chroma_results


"""
This function, get_response(query, meta_data), is designed to interact with the OpenAI GPT model using the ChatCompletion API.
It takes a user query and metadata as inputs, and returns a response generated by the model.

Parameters:
- query (str): The user's input query or prompt for the model.
- meta_data (list): A list containing relevant metadata information.
  - meta_data[0] (str): The API key to authenticate and access the OpenAI services.
  - meta_data[1] (str): The name or identifier of the specific GPT model to use.

Returns:
- str: The generated response text based on the user query and model's completion.

Note:
- Before calling this function, ensure that the OpenAI library is properly imported and installed.
- The function sets the API key and sends a user message to the GPT model for generating a response.
- The 'temperature' parameter controls the randomness of the response. Lower values (e.g., 0.1) make the output more deterministic, while higher values introduce more randomness.
- The function extracts the content of the generated response from the API response object.
"""

def get_response(query,meta_data):
    preamble = "<s>[INST] <<SYS>>" + meta_data[2] + "<</SYS>>\n"
    input_prompt = preamble + query + "[/INST]\n"
    sequences = pipeline(
    input_prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=2000
    )
    for seq in sequences:
        model_out = seq['generated_text'].split("[/INST]")[-1].strip()
    return model_out    


"""
    Extracts bulleted list items from the given text.

    Args:
        text (str): The input text containing a bulleted list.

    Returns:
        list: A list of sanitized bulleted list items without the bullet points
              and leading spaces.
"""
def get_search_terms(text):
    pattern = r'^[-•] (.*)?$'  # Assumes bullet points start with a hyphen and a space
    bulleted_list = re.findall(pattern, text, re.MULTILINE)
    return bulleted_list


"""
    Finds the first recommended title that has not been read yet.

    Args:
        read_titles (list): A list of strings representing titles that have already been read.
        recommended_titles (list): A list of strings representing titles recommended to read.

    Returns:
        str: The first title from recommended_titles that has not been read, or "All done!" if all titles
             from recommended_titles have been read.
"""
def find_next_unread_title(read_titles, recommended_titles):
    for title in recommended_titles:
        if title not in read_titles:
            return title
    return "All done!"


"""
Retrieves search results from Wikipedia API based on a query.

Args:
    query (str): The search query to be used for retrieving results.
    num_results (int): The number of search results to retrieve.

Returns:
    list: A list of dictionaries containing search result information.
"""
def get_wikipedia_search_results(query, num_results):
    base_url = "https://en.wikipedia.org/w/api.php"
    
    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": num_results
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    return data["query"]["search"]

"""
Retrieves the introductory text content of a Wikipedia page based on its title.

Args:
    title (str): The title of the Wikipedia page to retrieve content for.

Returns:
    str: The plain text introductory content of the specified Wikipedia page.
"""
def get_wikipedia_page_content(title):
    base_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "titles": title,
        "explaintext": True,  # Get plain text content
        "redirects": True     # Follow redirects
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    content = page.get("extract", "")

    return content



what_i_learned = ""
search_terms = [current_search_term]
seen = []


for i in range(1,num_iterations): 
    #search
    search_results = get_wikipedia_search_results(current_search_term, num_results)
    
    #Get the titles and choose the one to look at
    titles = []
    for result in search_results:
        #print("=====" + result['title'] + "\n")
        titles.append(result['title'])
    first = find_next_unread_title(seen,titles)
    
    #now that we've chosen one, we've already seen it, so add it to the list
    seen.append(first)
    try:
        #Get the content
        page_content = get_wikipedia_page_content(first)
        page = wikipedia.page(first)
        #append to the what I learned list
        what_i_learned = what_i_learned + "\n\n=====" + first + "\n" + page_content

        #add to chroma
        add_to_chroma(page.content,first,collection)

        res = get_response(prompt + current_search_term,meta)
        meta = [tokenizer,pipeline,prompt_question]
        question = get_response(page_content,meta)
        print("====" + question + "\n")
        
        with open(question_file,'a',encoding='utf-8') as question_writer:
            question_writer.write("===" + question + "\n\n")
            chroma_results = "".join(collection.query(query_texts=[question],n_results=5)['documents'][0])
            meta = [tokenizer,pipeline,prompt_chroma]
            answer = get_response(chroma_results + "\nAnd the question is: " + question + " Answer the question now: ",meta)
            question_writer.write(answer + "\n\n")
        
        #meta = [tokenizer,pipeline,prompt_search]
        #new_terms = get_response(question,meta)
        #print(new_terms)
        meta = [tokenizer,pipeline,prompt]
        terms = get_search_terms(res)
        
        current_search_term = find_next_unread_title(search_terms,terms)
    except:
        print("ope! That didn't work")

#print(what_i_learned)
with open(filename,'w',encoding='utf-8') as writer:
    writer.write(what_i_learned)
