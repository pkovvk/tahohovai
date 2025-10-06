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
    """Отправляем prompt к HF Inference (text_generation). Возвращаем срезанный ответ.

    Замечание: если модель поддерживает multimodal inference через Inference API, можно допилить
    передачу изображения (но API и модель должны это поддерживать). Сейчас используем OCR->text.
    """
    if not prompt_text:
        return ""

    # Добавляем инструкции, чтобы модель отвечала коротко и по делу
    system_instr = (
        "Ты — помощник для решения школьных задач.\n"
        "Отвечай как можно короче и по делу. Никакой лишней воды.\n"
        "Если нужен только ответ — выдавай только ответ. Если требуется шаги — выдавай минимально необходимые шаги.\n"
        "Ответ должен быть на том же языке, что и вопрос."
    )

    full_prompt = system_instr + "\nЗадача:\n" + prompt_text

    call_kwargs = {"model": REPO_ID, "max_new_tokens": 512}
    if REVISION:
        call_kwargs["revision"] = REVISION

    try:
        # text_generation как awaitable в AsyncInferenceClient: первый аргумент — prompt
        result = await hf_client.text_generation(full_prompt, **call_kwargs)
    except Exception as e:
        logging.exception("Ошибка при обращении к HuggingFace Inference: %s", e)
        raise

    # Различные форматы ответа — приводим к строке
    # Обычно result может быть str, dict или объект с полем generated_text
    if isinstance(result, str):
        return result.strip()

    # если это список или dict — пробуем достать форму
    # случаи: [{'generated_text': '...'}] или {'generated_text': '...'}
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"].strip()
        return str(first).strip()

    if isinstance(result, dict):
        if "generated_text" in result:
            return result["generated_text"].strip()
        # иногда key 'text' или 'data' встречается
        for k in ("text", "data"):
            if k in result:
                return str(result[k]).strip()

    # fallback
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
        # короткая информационная вставка — можно убрать
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
    # Можно выставить proxy/timeout/другие параметры при необходимости
    executor.start_polling(dp, skip_updates=True)
