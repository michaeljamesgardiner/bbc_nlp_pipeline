import os
import argparse
from glob import glob
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from io_utils import save_to_json

load_dotenv()

categories = {
    'business': 'Mergers and Acquisitions, Economic Growth and Indicators, Corporate Finance and Governance, Industry Performance, Monetary Policy and Currency, Corporate Restructuring and Litigation, Global Trade and Finance, ',
    'entertainment': 'Film, Music, Theatre and Performance, Awards and Nominations, Celebrity News and Obituaries, Arts and Culture, Industry Trends and Business, ',
    'politics': 'Election Politics, Immigration and Asylum Policy, Economic Policy and Budget, Foreign Policy and International Relations, Law and Order, Government Accountability and Scandal, Social Policy and Welfare, ',
    'sport': 'Athletics, Football, Rugby, Tennis,',
    'tech': 'Cybersecurity, Consumer Electronics and Mobile Technology, Internet and Online Services, Gaming Industry, Intellectual Property and Copyright, Emerging Technologies, '
}

def prompt_generator(article, file_path, dirpath):
    category = os.path.basename(dirpath)
    subtopics = categories.get(category, "")
    return f"""
Of the following categories:

{subtopics}

Which does the following text from a news article belong? Only return the sub category, nothing else:

{article}
"""

def output_validation(llm, prompt, valid_subcategories):
    max_retries = 5
    for _ in range(max_retries):
        response = llm.complete(prompt)
        result = response.text.strip()
        result_lines = [line.strip() for line in result.split('\n') if line.strip()]
        if not result_lines:
            continue
        first_line = result_lines[0]
        if any(first_line in v for v in valid_subcategories.values()):
            return response
    return response

def process_articles(llm, input_dir, prompt_generator, result_key="classification"):
    results = {}
    txt_files = glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article = f.read()

            prompt = prompt_generator(article, file_path, os.path.dirname(file_path))
            response = output_validation(llm, prompt, categories)
            lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            folder_name = os.path.basename(os.path.dirname(file_path))
            results[f"{folder_name} - {os.path.basename(file_path)}"] = {result_key: lines}
            print(f"Processed {file_path}")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
    return results

def main():
    parser = argparse.ArgumentParser(description="classify BBC articles into subcategories using Groq LLM")
    parser.add_argument("--input_dir", type=str, default=os.getenv("INPUT_DIR"), help="input directory containing category subfolders of .txt articles")
    parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR"), help="directory to save output JSON file")
    parser.add_argument("--output_file", type=str, default="subcategory_classification.json", help="output JSON file name")
    parser.add_argument("--model", type=str, default="meta-llama/llama-4-maverick-17b-128e-instruct", help="GROQ LLM model to use")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env")

    llm = Groq(model=args.model, api_key=api_key)
    data = process_articles(llm, args.input_dir, prompt_generator)
    save_to_json(data, output_path)
    print(f"saved subcategory classifications to {output_path}")

if __name__ == "__main__":
    main()


