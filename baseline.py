import json
import requests
import os

# Настройки Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "my-qwen"

def call_ollama(prompt):
    """
    Вызов локальной Ollama через HTTP API.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"

def run_baseline(test_file):
    if not os.path.exists(test_file):
        print(f"Файл {test_file} не найден. Сначала запусти prepare_data.py")
        return

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    print(f"Запуск Baseline на {len(test_data)} примерах...")
    
    results = []
    for i, ex in enumerate(test_data):
        # Базовый промпт (Zero-shot)
        prompt = (
            "Ты — помощник по извлечению информации. "
            "Извлеки из текста все научные термины и их категории в формате 'Термин (Категория)'. "
            "Перечисли их через запятую. Ничего не объясняй.\n\n"
            f"Текст: {ex['text']}\n\n"
            "Ответ:"
        )
        
        prediction = call_ollama(prompt)
        results.append({
            "text": ex['text'],
            "target": ex['entities'],
            "prediction": prediction
        })
        if (i + 1) % 5 == 0:
            print(f"Обработано {i+1} из {len(test_data)}")

    # Оценка
    total_score = 0
    for res in results:
        target_list = [t.strip().lower() for t in res['target'].split(',') if t.strip()]
        if not target_list:
            total_score += 1
            continue
            
        pred_text = res['prediction'].lower()
        found = sum(1 for t in target_list if t in pred_text)
        total_score += found / len(target_list)

    avg_score = total_score / len(results)
    print(f"\n--- РЕЗУЛЬТАТ BASELINE ---")
    print(f"Средняя полнота (Recall): {avg_score:.2%}")
    print(f"Результаты сохранены в baseline_results.json")
    
    with open('baseline_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_baseline('test.jsonl')
