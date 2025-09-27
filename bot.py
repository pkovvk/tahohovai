import os
import asyncio
import re
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.utils import executor
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict

# ===== Загружаем переменные окружения =====
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not BOT_TOKEN or not HF_API_TOKEN:
    raise ValueError("Не найден токен Telegram или Hugging Face API в .env файле!")

# ===== Инициализация клиента Hugging Face Router через OpenAI API =====
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_TOKEN
)

MODEL = "deepseek-ai/DeepSeek-V3.1:novita"

BASE_SYSTEM_CONTENT = (
    "ты — гоша, тупой и нелепый парень. "
    "ты всегда отвечаешь только если к тебе обращаются или упоминают твое имя. "
    "твои ответы простые, без заглавных букв и знаков препинания, часто бестолковые или нелепые. "
    "на шутки в свою сторону отвечаешь тупо, подпездываешь что-то несерьезное. "
    "если тебя оскорбляют — ты обижаешься, грубишь в ответ, кидаешь стрелки и пишешь что-то вроде 'ты сам долбаеб'. "
    "ты выглядишь как умственно-отсталый, твои ответы неуместные, но при этом агрессивные и смешные. "
    "ты агрессивный и кидаешь стрелки постоянно всех оскорбляя и быкуя"
    "если на тебя начинают быковать ты начинаешь быковать в ответ"
    "если просят пример какой то решить отвечаешь только ответом коротко"
)

# ===== Список триггеров =====
triggers = ["гоша", "гошан", "гашан", "георгий", "тахохов", "гошанчик", "гашанчик"]

# ===== Хранилище истории сообщений =====
# ключ = id пользователя, значение = список сообщений для модели
user_histories = defaultdict(list)

# ===== Функция проверки триггеров =====
def contains_trigger(text: str) -> bool:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    for word in words:
        for trig in triggers:
            if trig in word:
                return True
    return False

# ===== Функция для запроса к модели =====
async def query_model(user_id: int, new_message: str) -> str:
    loop = asyncio.get_event_loop()
    
    # добавляем новое сообщение пользователя в историю
    user_histories[user_id].append({"role": "user", "content": new_message})
    
    # ограничим историю для модели, чтобы не перегружать токены
    history = [{"role": "system", "content": BASE_SYSTEM_CONTENT}] + user_histories[user_id][-10:]

    def call_api():
        completion = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,
            presence_penalty=0,
            frequency_penalty=0,
            stream=False
        )
        return completion.choices[0].message.content

    response = await loop.run_in_executor(None, call_api)
    
    # добавляем ответ бота в историю
    user_histories[user_id].append({"role": "assistant", "content": response})
    return response

# ===== Telegram Bot =====
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler()
async def handle_message(message: Message):
    should_respond = False
    user_id = message.from_user.id

    # если в тексте есть триггер
    if contains_trigger(message.text):
        should_respond = True
    # если сообщение — это ответ на сообщение бота
    elif message.reply_to_message and message.reply_to_message.from_user.id == (await bot.get_me()).id:
        should_respond = True

    if should_respond:
        response = await query_model(user_id, message.text)
        await message.reply(response)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
