import os
import logging
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import asyncio

# -------------------- Настройки и логирование --------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
MODEL_REV = os.getenv("MODEL_REV", "hyperbolic")
HF_PROVIDER = os.getenv("HF_PROVIDER", "hyperbolic")

if not BOT_TOKEN:
    raise RuntimeError("Переменная окружения BOT_TOKEN не установлена")
if not HF_TOKEN:
    raise RuntimeError("Переменная окружения HF_TOKEN не установлена")

# Telegram и Hugging Face
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

hf_client = InferenceClient(
    provider=HF_PROVIDER,
    api_key=HF_TOKEN,
)

# -------------------- Локальный lock для очереди --------------------
hf_lock = asyncio.Lock()

# -------------------- Функция запроса к модели --------------------
async def ask_model(prompt_text: str = "", image_url: str = None) -> str:
    """Отправка текста и (опционально) изображения через HuggingFace Hub, ответ на русском и кратко."""
    system_message = {
        "role": "system",
        "content": [
            {"type": "text", "text": "Ты говоришь только на русском языке и отвечаешь коротко и по делу."}
        ]
    }

    user_content = []
    if prompt_text:
        user_content.append({"type": "text", "text": prompt_text})
    if image_url:
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})

    async with hf_lock:  # один запрос в момент времени
        try:
            def sync_call():
                return hf_client.chat.completions.create(
                    model=f"{MODEL_NAME}:{MODEL_REV}",
                    messages=[system_message, {"role": "user", "content": user_content}],
                    max_tokens=256,
                    temperature=0.2
                )

            completion = await asyncio.to_thread(sync_call)

            # Извлекаем текст
            text = completion.choices[0].message
            if isinstance(text, str):
                return text.strip()
            elif hasattr(text, "content"):
                content = text.content
                if isinstance(content, list):
                    return " ".join(c["text"] for c in content if c["type"] == "text").strip()
                return str(content)
            return str(text)

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

    caption = message.caption or "Опиши изображение."
    answer = await ask_model(caption, image_url=file_url)
    await message.reply(answer or "Модель не вернула ответ.")

# -------------------- Запуск --------------------
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
