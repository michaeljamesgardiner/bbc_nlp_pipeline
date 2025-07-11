import os
import argparse
from glob import glob
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from io_utils import save_to_txt

load_dotenv()

categories = ["business", "entertainment", "politics", "sport", "tech"]
max_chunk_chars = 20000  # approximately 5000 tokens

def concatenate_articles_with_delimiter(input_dir):
    articles = []
    txt_files = glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                articles.append(f.read().strip())
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
    return articles

def chunk_articles(articles, max_chars):
    chunks = []
    current_chunk = ""
    for article in articles:
        article_with_delim = "\n\n===new_article\n\n" + article.strip()
        if len(current_chunk) + len(article_with_delim) <= max_chars:
            current_chunk += article_with_delim
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = article_with_delim
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def extract_topics_from_chunk(llm, chunk_text, category):
    prompt = f"""
Each {category} news article in the text below is delineated by ===new_article.

For each article, return a {category} news article subtopic. Return only the subtopic and nothing else.

{chunk_text}
"""
    response = llm.complete(prompt)
    return response.text.strip()

def condense_categories(llm, responses, category):
    prompt = f"""
From the following list, return a unique list of {category} news article subtopics.

Condense them into a maximum of 7 broad categories (can be fewer). For each:
- State the number of articles that fall into that category
- Provide a short explanation of the category

Finally, list any articles that did not fit into any category.

{responses}
"""
    response = llm.complete(prompt)
    return response.text.strip()

def main():
    parser = argparse.ArgumentParser(description="topic modelling for BBC articles by category using GROQ LLM")
    parser.add_argument("--input_dir", type=str, default=os.getenv("INPUT_DIR"), help="input directory containing category subfolders of .txt articles")
    parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR"), help="directory to save topic summaries")
    parser.add_argument("--model", type=str, default="meta-llama/llama-4-maverick-17b-128e-instruct", help="GROQ LLM model to use")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env")

    llm = Groq(model=args.model, api_key=api_key)

    for category in categories:
        print(f"processing category - {category}")
        input_path = os.path.join(args.input_dir, category)
        output_path = os.path.join(args.output_dir, f"{category}_topic_summary.txt")

        articles = concatenate_articles_with_delimiter(input_path)
        chunks = chunk_articles(articles, max_chunk_chars)
        print(f"loaded {len(articles)} articles into {len(chunks)} chunks")

        all_responses = []
        for i, chunk in enumerate(chunks):
            print(f"extracting topics from chunk {i + 1}")
            chunk_response = extract_topics_from_chunk(llm, chunk, category)
            all_responses.append(chunk_response)

        combined_responses = "\n".join(all_responses)
        print("condensing topic categories")
        final_summary = condense_categories(llm, combined_responses, category)

        save_to_txt(final_summary, output_path)
        print(f"saved topic summary to {output_path}")

if __name__ == "__main__":
    main()
