import os
import csv
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

pipe = pipeline("text-classification", model="pkovvk/tahohovai")
tokenizer = AutoTokenizer.from_pretrained("pkovvk/tahohovai")
model = AutoModelForSequenceClassification.from_pretrained("pkovvk/tahohovai")

load_dotenv()  # загружаем переменные из .env
TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# Массив слов-триггеров
TRIGGERS = ["тахохов", "гоша", "гоши", "гошан", "гашан", "георгий"]
QUESTION_TRIGGERS = ["гоша", "гошан", "гашан", "георгий", "тахохов"]

# Стикер
STICKER_ID = "CAACAgIAAxkBAAEPck5o1mg46qdyJBIQ5VRWGa63LF_SoAAC9YAAAmsuuUocW-dcBOCJVDYE"
STICKER_YES = "CAACAgIAAxkBAAEPc0do13Zr1ZlRPDALYuThmK_ZVvmhyAACiHoAArXWwErEbbppkzsfxjYE"
STICKER_NO = "CAACAgIAAxkBAAEPc0Vo13Zj1baaHb_brXXDmBcFR8KD-AACf3sAArC_wErbTwVtSWfALjYE"

# Ответ в личке
PRIVATE_REPLY = "не пиши мне"

UNKNOWN_CSV = "dataset.csv"

def save_unknown(question: str):
    file_exists = os.path.isfile(UNKNOWN_CSV)
    with open(UNKNOWN_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["question"])  # заголовок
        writer.writerow([question])

@dp.message_handler(content_types=types.ContentType.TEXT)
async def check_messages(message: types.Message):
    isQuestion = False
    text = message.text.lower()
    
    if message.chat.type == "private":
        await message.answer(PRIVATE_REPLY)
        return
    
    # Проверка на вопрос
    for trigger in QUESTION_TRIGGERS:
        if text.startswith(trigger) and text.endswith("?"):
            isQuestion = True
            # Убираем триггерное слово из начала
            question = text[len(trigger):].strip()
            # Получаем классификацию от модели
            result = pipe(question)  # [{'label': 'yes', 'score': 0.95}, ...]
            answer = result[0]['label']  # берем 'yes', 'no' или 'unknown'
            if answer == "yes":
                await message.reply_sticker(STICKER_YES)
            elif answer == "no":
                await message.reply_sticker(STICKER_NO)
            elif answer == "unknown":
                save_unknown(question)
                await message.reply("надо подумать")
            else:
                await message.reply("error")
            break
        
    if any(word in text for word in TRIGGERS):
        if isQuestion == False:
            await message.reply_sticker(STICKER_ID)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
