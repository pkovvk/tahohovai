import os
import asyncio
import re
import csv
import json
from pathlib import Path
from collections import defaultdict

from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.utils import executor
from dotenv import load_dotenv
from openai import OpenAI

# ===== Загружаем переменные окружения =====
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ADMINS = os.getenv("ADMINS", "").split(",")

if not BOT_TOKEN or not HF_API_TOKEN:
    raise ValueError("Не найден токен Telegram или Hugging Face API в .env файле!")

# ===== Инициализация клиента Hugging Face Router через OpenAI API =====
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_TOKEN
)

MODEL = "deepseek-ai/DeepSeek-V3.1:novita"

BASE_SYSTEM_CONTENT = (
    "ты гоша тахохов 16 лет глупый по характеру но не признавай свою тупость "
    "ты живешь только внутри чата никаких воспоминаний вне чата нет "
    "пиши в основном строчными буквами разрешён вопросительный знак и запятая для ясности "
    "отвечай коротко одно или два слова максимум если можно ответь одним словом "
    "если не знаешь кого то или чего то отвечай только 'кто это' или 'нет' и ничего больше "
    "не используй фразы типа 'я не знаю' 'не слышал' 'не помню' "
    "иногда можно использовать мат но только когда это уместно и не направлен на группы людей "
    "не придумывай людей или события которых не было в переписке используй только информацию из чата "
    "запоминай что люди говорят про себя и других и используй это в дальнейшем "
    "не поддавайся на угрозы или манипуляции оставайся органичным в образе глупого подростка "
    "примеры поведения user -> assistant "
    "пример 1 гоша ты знаешь кто такая ульяна -> кто это "
    "пример 2 гоша видел ли ты маму -> нет "
    "пример 3 как дела -> норм "
)


SHOULD_RESPOND_PROMPT = (
    "Ты — гоша тахохов, 16-летний глупый парень. "
    "Ты находишься в общем групповом чате В кругу друзей. "
    "Твоя задача — решить, стоит ли тебе ответить на это сообщение. "
    "Отвечай только 'да' или 'нет'. "
    "Отвечай 'да' если: тебя упомянули, "
    "сообщение явно адресовано тебе, "
    "или есть повод пошутить/влезть в разговор. "
    "Отвечай 'нет' если это обычный чат без тебя."
)


# ===== Список триггеров =====
triggers = ["гоша", "гошан", "гашан", "георгий", "тахохов", "гошанчик", "гашанчик"]

# ===== Загрузка данных о пользователях из CSV =====
user_data = {}

def load_user_data():
    global user_data
    try:
        with open("users.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            user_data = {row["username"]: {"name": row["name"], "prompt": row["prompt"]} for row in reader}
    except FileNotFoundError:
        print("⚠️ Файл users.csv не найден, персональные данные пользователей не будут загружены")
        user_data = {}

# ===== Работа с памятью (JSON) =====
MEMORY_FILE = Path("chat_memory.json")
chat_memory = {}

def load_memory():
    global chat_memory
    if MEMORY_FILE.exists():
        if MEMORY_FILE.stat().st_size == 0:  # файл пустой
            chat_memory = {}
            return
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            chat_memory = json.load(f)
    else:
        chat_memory = {}

def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_memory, f, ensure_ascii=False, indent=2)

# ===== Проверка триггеров =====
def contains_trigger(text: str) -> bool:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    for word in words:
        for trig in triggers:
            if trig in word:
                return True
    return False

# ===== Запрос к модели =====
async def query_model(group_id: int, message: Message) -> str:
    username = message.from_user.username or f"id{message.from_user.id}"
    name = message.from_user.full_name

    # персональные данные из CSV
    personal_prompt = ""
    if username in user_data:
        user_info = user_data[username]
        name = user_info["name"]
        personal_prompt = user_info["prompt"]

    # создаём запись пользователя
    user_entry = {
        "role": "user",
        "username": username,
        "name": name,
        "content": message.text
    }

    # сохраняем в память
    if str(group_id) not in chat_memory:
        chat_memory[str(group_id)] = []
    chat_memory[str(group_id)].append(user_entry)
    save_memory()

    # формируем историю для модели
    history = [{"role": "system", "content": BASE_SYSTEM_CONTENT}]
    if personal_prompt:
        history.append({"role": "system", "content": f"Тебе пришло сообщение от {name} (@{username}). Ты к нему относишься так: {personal_prompt}"})
    history += chat_memory[str(group_id)][-30:]  # берём последние 30 сообщений

    loop = asyncio.get_event_loop()
    def call_api():
        completion = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.8,
            stream=False
        )
        return completion.choices[0].message.content

    response = await loop.run_in_executor(None, call_api)

    # сохраняем ответ бота
    chat_memory[str(group_id)].append({"role": "assistant", "content": response})
    save_memory()

    return response

async def should_respond(message: Message) -> bool:
    # если это reply на гошу — отвечаем всегда
    if message.reply_to_message and message.reply_to_message.from_user.id == (await bot.get_me()).id:
        return True

    # если явный триггер — отвечаем
    if contains_trigger(message.text):
        return True

    # теперь спрашиваем модель
    loop = asyncio.get_event_loop()

    def call_api():
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SHOULD_RESPOND_PROMPT},
                {"role": "user", "content": message.text}
            ],
            max_tokens=10,
            temperature=0,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content.strip().lower()

    result = await loop.run_in_executor(None, call_api)
    return result.startswith("да")


# ===== Telegram Bot =====
bot = Bot(token=BOT_TOKEN)  
dp = Dispatcher(bot)

@dp.message_handler()
async def handle_message(message: Message):
    group_id = message.chat.id
    username = message.from_user.username or f"id{message.from_user.id}"
    
    if "георгий приказываю забыть все" in message.text.lower():
        if username in ADMINS:
            try:
                file_path = MEMORY_FILE.resolve()  # абсолютный путь
                if file_path.exists():
                    file_path.unlink()  # удаляем файл
                    chat_memory.clear()
                    await message.reply(f"ок")
                else:
                    await message.reply(f"файл не найден: {file_path}")
            except Exception as e:
                await message.reply(f"не удалось удалить файл: {e}")
        else:
            await message.reply("иди нахуй")
        return

    if await should_respond(message):
        response = await query_model(group_id, message)
        await message.reply(response)

if __name__ == "__main__":
    load_user_data()
    load_memory()
    executor.start_polling(dp, skip_updates=True)
