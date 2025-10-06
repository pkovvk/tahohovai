import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from openai import AsyncOpenAI

# -------------------- Настройки и логирование --------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct:novita")

if not BOT_TOKEN:
    raise RuntimeError("Переменная окружения BOT_TOKEN не установлена")
if not HF_TOKEN:
    raise RuntimeError("Переменная окружения HF_TOKEN не установлена")

# Telegram и Hugging Face (через OpenAI-совместимый API)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

hf_client = AsyncOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# -------------------- Функция запроса к модели --------------------
async def ask_model(prompt_text: str = "", image_url: str = None) -> str:
    """Отправка текста и (опционально) изображения в Qwen3-VL."""
    content = []
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})
    if image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    try:
        completion = await hf_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
            temperature=0.2
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.exception("Ошибка при запросе к модели: %s", e)
        return f"Ошибка при запросе к модели: {e}"

# -------------------- Хэндлеры Telegram --------------------
@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    await message.reply("Обрабатываю...")
    answer = await ask_model(message.text)
    await message.reply(answer or "Модель не вернула ответ.")

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    await message.reply("Секунду, анализирую фото...")
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"

    # Передаем в модель URL фото и текст, если он есть
    caption = message.caption or "Опиши изображение."
    answer = await ask_model(caption, image_url=file_url)
    await message.reply(answer or "Модель не вернула ответ.")

# -------------------- Запуск --------------------
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
