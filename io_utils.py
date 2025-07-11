import json

def save_to_json(data, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"saved to {output_path}")
    except Exception as e:
        print(f"failed to save {e}")

def save_to_txt(data, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"saved to {output_path}")
    except Exception as e:
        print(f"failed to write file {e}")
