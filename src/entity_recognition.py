import os
import argparse
from glob import glob
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from io_utils import save_to_json

load_dotenv()

prompt_list = {
    "media_personalities": {
        "prompt": (
            "return a list, for every distinct named human person within the text, return the name followed the string ' --- ', followed by that person's occupation. "
            "Separate each named human person in the list with a line break. Strictly return only this, in this format, with no extra commentary or introduction or a heading. Return nothing else.\n\n"
        ),
        "filename": "media_personalities.json",
        "key": "entities"
    },
    "april_events": {
        "prompt": (
            "return a list, for every distinct event within the text occurring in April, and ONLY in April, return a summary of the event followed by the string ' --- ', followed by any information about the date or when in April the event occurred. "
            "Separate each event in the list with a line break. Strictly return only this, in this format, with no extra commentary or introduction or a heading. Return nothing else. If the event does not occur in April, return nothing.\n\n"
        ),
        "filename": "summary_of_events_in_april.json",
        "key": "events"
    }
}

def process_articles(llm, input_dir, base_prompt, result_key="result"):
    results = {}
    txt_files = glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article = f.read()

            prompt = base_prompt + article
            response = llm.complete(prompt)
            lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]

            # Output validation: keep lines with '---' and both sides non-empty
            lines = [line for line in lines if '---' in line]
            lines = [line for line in lines if all(part.strip() for part in line.split('---'))]
            # Remove duplicates by name
            lines = list({line.split(' --- ')[0].strip(): line for line in lines}.values())

            if lines:
                folder_name = os.path.basename(os.path.dirname(file_path))
                results[f"{folder_name} - {os.path.basename(file_path)}"] = {result_key: lines}
                print(f"Processed {file_path}")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description="extract media personalities or april events from BBC articles using GROQ LLM")
    parser.add_argument("--input_dir", type=str, default=os.getenv("INPUT_DIR"), help="input directory containing category subfolders of .txt articles")
    parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR"), help="output JSON file name")
    parser.add_argument("--mode", choices=prompt_list.keys(), required=True, help="choose either 'media_personalities' or 'april_events'")
    parser.add_argument("--model", type=str, default="meta-llama/llama-4-maverick-17b-128e-instruct", help="GROQ LLM model to use")

    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env")

    llm = Groq(model=args.model, api_key=api_key)

    prompt_info = prompt_list[args.mode]
    base_prompt = prompt_info["prompt"]
    result_key = prompt_info["key"]
    output_file = os.path.join(args.output_dir, prompt_info["filename"])

    data = process_articles(llm, args.input_dir, base_prompt, result_key=result_key)
    save_to_json(data, output_file)

if __name__ == "__main__":
    main()
