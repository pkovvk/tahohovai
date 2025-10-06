import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from huggingface_hub import AsyncInferenceClient

# -------------------- Настройки и логирование --------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
RAW_MODEL = os.getenv("MODEL", "").strip()  # ожидaется owner/repo или owner/repo:revision
MODEL_REV = os.getenv("MODEL_REV", "").strip()  # опционально можно задать ревизию в отдельной переменной

if not BOT_TOKEN:
    raise RuntimeError("Переменная окружения BOT_TOKEN не установлена")
if not HF_TOKEN:
    raise RuntimeError("Переменная окружения HF_TOKEN не установлена")
if not RAW_MODEL:
    raise RuntimeError("Переменная окружения MODEL не установлена (например: owner/repo или owner/repo:revision)")

# если ревизия указана в MODEL через ':' — разберём
if ":" in RAW_MODEL:
    REPO_ID, REVISION = RAW_MODEL.split(":", 1)
else:
    REPO_ID = RAW_MODEL
    REVISION = MODEL_REV or None

# Инициализация Telegram и HF клиентов
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
hf_client = AsyncInferenceClient(token=HF_TOKEN)

# -------------------- Функции OCR и вызова модели --------------------

def extract_text_from_image(file_path: str) -> str:
    """Извлекает текст из изображения через pytesseract (rus+eng)."""
    try:
        img = Image.open(file_path)
    except Exception as e:
        logging.exception("Не удалось открыть изображение: %s", e)
        return ""
    try:
        # указываем языки, если они установлены в tesseract (rus, eng)
        text = pytesseract.image_to_string(img, lang='rus+eng')
    except Exception:
        # fallback без указания языков
        text = pytesseract.image_to_string(img)
    return text.strip()


async def ask_model(prompt_text: str) -> str:
    """Используем chat completion (conversational) для моделей, которые поддерживают conversational task.

    Меняем подход: вместо text_generation вызываем chat.completions.create у AsyncInferenceClient.
    """
    if not prompt_text:
        return ""

    # Система — коротко, чтобы модель отвечала без воды
    system_instr = (
        "Ты — помощник для решения школьных задач."
        "Отвечай как можно короче и по делу — только финальный ответ без лишней воды."
    )

    messages = [
        {"role": "system", "content": system_instr},
        {"role": "user", "content": prompt_text},
    ]

    call_kwargs = {
        "model": REPO_ID,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.0,
    }

    # note: AsyncInferenceClient.chat.completions.create is awaited
    try:
        result = await hf_client.chat.completions.create(**call_kwargs)
    except Exception as e:
        logging.exception("Ошибка при обращении к HuggingFace Inference: %s", e)
        raise

    # Попробуем корректно извлечь текст ответа из разных форматов
    try:
        # result.choices is usual
        choices = getattr(result, "choices", None) or (result.get("choices") if isinstance(result, dict) else None)
        if choices:
            first = choices[0]
            # try object attribute
            msg = getattr(first, "message", None) or (first.get("message") if isinstance(first, dict) else None)
            # if message is a dict with 'content'
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    # list of {'type':'text','text':'...'} or strings
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                        elif isinstance(item, str):
                            parts.append(item)
                    return "".join(parts).strip()
            # sometimes choice.message is a simple string under 'content' or 'text'
            if isinstance(first, dict):
                # check several possible keys
                for key in ("text", "content", "message"):
                    val = first.get(key)
                    if isinstance(val, str):
                        return val.strip()
            # try attribute .message.content
            if hasattr(first, "message"):
                m = first.message
                if isinstance(m, str):
                    return m.strip()
                if isinstance(m, dict):
                    c = m.get("content")
                    if isinstance(c, str):
                        return c.strip()
            # fallback to string of first
            return str(first).strip()
    except Exception:
        logging.exception("Не удалось распарсить структуру ответа, возвращаю сырые данные")
        return str(result).strip()

    # ultimate fallback
    return str(result).strip()


# -------------------- Хэндлеры Telegram --------------------

@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    await message.reply("Обрабатываю задачу...")
    try:
        answer = await ask_model(message.text)
    except Exception as e:
        await message.reply(f"Ошибка при обращении к HF API: {e}")
        return
    if not answer:
        await message.reply("Модель вернула пустой ответ.")
        return
    await message.reply(answer)


@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    await message.reply("Распознаю и решаю задачу с фото...")
    photo = message.photo[-1]
    file_path = f"/tmp/tg_img_{message.message_id}.jpg"
    try:
        await photo.download(file_path)
        text = extract_text_from_image(file_path)
        if not text:
            await message.reply("Не удалось распознать текст на изображении.")
            return
        logging.info("OCR text: %s", text[:200])
        try:
            answer = await ask_model(text)
        except Exception as e:
            await message.reply(f"Ошибка при обращении к HF API: {e}")
            return
        if not answer:
            await message.reply("Модель вернула пустой ответ.")
            return
        await message.reply(answer)
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


# -------------------- Запуск --------------------
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
