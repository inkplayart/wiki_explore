#A fun little llama 2 chatbot. Learn more here: https://huggingface.co/blog/llama2

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from collections import deque
import chromadb

#You cannot run this script unless you have a ./chroma_db folder AND it has a useful databse in it. Run this after "wiki_explore_llama_post.py"
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="wiki")

model= "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

#Whether this is absolutely necessary is a matter for debate, but I'm following the hf tutorial
systemPrompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.  

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

You will be provided with a context and a query. Base your answer to the query on the context as much as possible. If the answer is not in the context, say so. Do not make up an answer.
<</SYS>>"""

#Get the query from the user
query = input("USER: >> ")

#This is the context. I chose a maxlen of 8, but honestly this should be based on the tokens, not just some random string length
context = deque(maxlen=8) #keep it small so it stops getting confused and to prevent very long chain reasoning
context.append(query + "[/INST]") #we keep appending to the queue, and the oldest stuff pops off for us
while (query != "EXIT"):
    if(query == "RESET"): #If you are getting weird answers, just type RESET and the context initializes back to empty
        context.clear()
    else:
        #Generate the text. We are prompting with the system prompt plus the entire context
        
        chroma_results = "".join(collection.query(query_texts=[query],n_results=5)['documents'][0])
        #print(chroma_results)
        sequences = pipeline(
            systemPrompt + "The context is: " + chroma_results + " \nThe query is: " + query + "\nRemember to answer the query using the context. [/INST]",
            max_new_tokens=512, #Change this if you want, but the bigger it is the more the context queue gets screwed
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        #We split the output of the model on the [/INST] string, and take the last element of the resulting list, which
        #gives us the last output of the model
        model_out = ""
        for seq in sequences:
            model_out = seq['generated_text'].split("[/INST]")[-1].strip()
        print("\n\n=====\nLLAMA: \n=====" + model_out + "\n")
        
        context.append(model_out + "</s><s>[INST]") #all of this prompting tag stuff is from the hf blog post
        
    query = input("USER: >> ")
    context.append(query + "[/INST]")



