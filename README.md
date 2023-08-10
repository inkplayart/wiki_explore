# wiki_explore
Explore wikipedia, query it using a vector DB!

## Installation

You need transformers, pytorch, chromadb and wikipedia, at least

In the same folder as the code, create a directory called `chroma_db`. This is where the vector DB will persist.

## Step 1: Ingest Wikipedia

Start by modifying wiki_explore_llama_post.py. Line 1 has the current search term. This is the starting point, and it should be a regular Wikipedia search query. Line 29 is the name of the output file. This is a temporary file, so it can be overwritten. It's more of a log than anything. question_file is a cool file to read. It contains questions that the AI asked as it traversed Wikipedia, and then its best attempts to answer them based on the current state of the vector DB.

Once you've changed these things, run the script. num_iterations of about 200 is the max I've gotten it to run for before it starts to error out. I think the problem is that eventually the search terms get re-used so much that everything is the same? I don't know, and honestly idgaf. Fix it if you want and submit a PR!

## Step 2: Talk to Wikipedia

The chat interface lives in explorer_post.py. Run it after you've done your ingestion and ask questions. There is no memory in the chat interface, although in theory you can add it in. Again, that's up to you.

## Step 3: Re-ingest

If you run it on a new starting term, it will append new information into the databse. IN THEORY this will make it smarter, but...I haven't tested it with too many queries. I expect that at some point the vectors will be semantically so similar that it will degrade the quality, and answers will make no sense. Let me know what happens.

## You probably want to keep backups of your DB

Just copy the chroma_db folder into some other backups folder, maybe rename it (like "chroma_db_only_biology"). That way if you ever get your DB too big and it goes off the rails, you can just walk it back to a working version. I don't know, you do you.
