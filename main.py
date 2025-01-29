import os
import json
import argparse
import nltk
import torch
import deepl
import whisper

DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "YOUR_DEEPL_API_KEY")
translator = deepl.Translator(DEEPL_API_KEY)


# GPU/MPS 사용 여부 확인
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# argparse 설정
parser = argparse.ArgumentParser(description="Whisper STT: Convert speech to text from an audio file.")
parser.add_argument("input_file", type=str, help="Path to the input audio file (e.g., file.m4a)")
args = parser.parse_args()

def transcribe(input_file):
    output_file = os.path.splitext(input_file)[0] + ".txt"
    model = whisper.load_model("large", device=device)
    result = model.transcribe(input_file, language="en")
    text = result['text']

    # 텍스트 파일 저장
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


def translate(input_file):
    # 파일 읽기
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # NLTK 다운로드 (문장 단위 분리를 위한 tokenizer)
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    translated_data = list()

    for sentence in sentences:
        print(sentence, translator.translate_text(sentence, target_lang="KO"))
        translated_data.append({"en": sentence, "ko": translator.translate_text(sentence, target_lang="KO")})

    output_file = os.path.splitext(input_file)[0] + ".json"
    # JSON 파일로 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # transcribe(args.input_file)
    translate('/Users/evan/Documents/Airbus_Amber/transcribed_text.txt')
