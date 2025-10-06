import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient

# -------------------- Настройки и логирование --------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = os.getenv("MODEL", "").strip()  # ожидается owner/repo (Qwen/Qwen3-VL-235B-A22B-Instruct)

if not BOT_TOKEN:
    raise RuntimeError("Переменная окружения BOT_TOKEN не установлена")
if not HF_TOKEN:
    raise RuntimeError("Переменная окружения HF_TOKEN не установлена")
if not MODEL:
    raise RuntimeError("Переменная окружения MODEL не установлена (например: Qwen/Qwen3-VL-235B-A22B-Instruct)")

# Инициализация Telegram и HF клиентов
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
hf_client = AsyncInferenceClient(token=HF_TOKEN)

# -------------------- Функция вызова модели --------------------

async def ask_model(prompt_text: str = "", image_path: str = None) -> str:
    """Отправляем в модель текст и/или изображение, получаем ответ."""
    if not prompt_text and not image_path:
        return "Нет текста или изображения для обработки."

    # Система — коротко, чтобы модель отвечала по делу
    system_instr = (
        "Ты — помощник для решения школьных задач. "
        "Отвечай только финальным результатом без воды. "
        "Математика, физика, химия, графики — обрабатывай корректно."
    )

    messages = [
        {"role": "system", "content": system_instr},
    ]
    if prompt_text:
        messages.append({"role": "user", "content": prompt_text})

    # Подготовка аргументов для chat.completions.create
    call_kwargs = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.0,
    }

    if image_path:
        with open(image_path, "rb") as f:
            call_kwargs["image"] = f.read()  # Qwen3-VL-Instruct умеет принимать image

    try:
        result = await hf_client.chat.completions.create(**call_kwargs)
    except Exception as e:
        logging.exception("Ошибка при обращении к HuggingFace Inference: %s", e)
        return f"Ошибка при обращении к HF API: {e}"

    # Извлекаем текст из ответа
    try:
        choices = getattr(result, "choices", None) or (result.get("choices") if isinstance(result, dict) else None)
        if choices:
            first = choices[0]
            msg = getattr(first, "message", None) or (first.get("message") if isinstance(first, dict) else None)
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    return "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in msg]).strip()
            if isinstance(first, dict):
                for key in ("text", "content"):
                    val = first.get(key)
                    if isinstance(val, str):
                        return val.strip()
            if hasattr(first, "message") and isinstance(first.message, str):
                return first.message.strip()
        return str(result).strip()
    except Exception:
        logging.exception("Не удалось распарсить ответ модели, возвращаю сырые данные")
        return str(result).strip()

# -------------------- Хэндлеры Telegram --------------------

@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    await message.reply("Обрабатываю задачу...")
    answer = await ask_model(prompt_text=message.text)
    await message.reply(answer or "Модель вернула пустой ответ.")

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    await message.reply("Распознаю и решаю задачу с фото...")
    photo = message.photo[-1]
    file_path = f"/tmp/tg_img_{message.message_id}.jpg"
    try:
        await photo.download(destination_file=file_path)
        answer = await ask_model(image_path=file_path)
        await message.reply(answer or "Модель вернула пустой ответ.")
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

# -------------------- Запуск --------------------
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
