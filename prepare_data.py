import json
import random
import os

# Путь к данным внутри проекта
INPUT_FILE = os.path.join('approach-to-IE-based-on-a-chain-of-LLMs', 'ents', 'doc', 'articles_dataset.json')

def prepare_ner_data(input_file, train_size=50, test_size=50):
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не найден!")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_examples = []
    for article in data['articles']:
        for art_id, fragments in article.items():
            for frag in fragments:
                text = frag['text'].strip()
                if not text: continue
                entities = sorted(list(set([f"{t['value']} ({t['class']})" for t in frag.get('terms', [])])))
                all_examples.append({"text": text, "entities": ", ".join(entities)})
    
    random.seed(42)
    random.shuffle(all_examples)
    
    def save_jsonl(filename, data_list):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    save_jsonl('train.jsonl', all_examples[:train_size])
    save_jsonl('test.jsonl', all_examples[train_size:train_size + test_size])
    print(f"Готово! Сохранено в {os.getcwd()}")

if __name__ == "__main__":
    prepare_ner_data(INPUT_FILE)
