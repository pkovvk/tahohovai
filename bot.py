import os
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from huggingface_hub import AsyncInferenceClient

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
MODEL = os.getenv("MODEL")  # например "deepseek-ai/DeepSeek-V3.1-Terminus:novita"
HF_TOKEN = os.getenv("API_TOKEN")

if not BOT_TOKEN or not HF_TOKEN or not MODEL:
    raise RuntimeError("Установи переменные окружения BOT_TOKEN, HF_TOKEN и MODEL в .env")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# Асинхронный клиент — не блокирует aiogram event loop
hf_client = AsyncInferenceClient(token=HF_TOKEN)

def extract_text_from_image(file_path):
    img = Image.open(file_path)
    # рус+англ, можно добавить другие языки, если установлены в tesseract
    text = pytesseract.image_to_string(img, lang='rus+eng')
    return text.strip()

async def ask_model(prompt_text):
    """
    Вызываем асинхронный text_generation. Передаём prompt как позиционный аргумент.
    Возвращаем строку.
    """
    # Подставь дополнительные параметры генерации при необходимости:
    # max_new_tokens, temperature и т.п.
    try:
        result = await hf_client.text_generation(prompt_text, model=MODEL, max_new_tokens=1024)
    except Exception as e:
        # Пробрасываем понятную ошибку наверх
        raise

    # result обычно возвращает строку, если details=False (по умолчанию)
    if isinstance(result, str):
        return result.strip()
    # В некоторых случаях возвращается объект TextGenerationOutput
    # пытаемся достать текст из атрибута
    generated = getattr(result, "generated_text", None)
    if generated:
        return generated.strip()
    # fallback: str(result)
    return str(result).strip()

@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    await message.reply("Обрабатываю задачу...")
    try:
        answer = await ask_model(message.text)
    except Exception as e:
        await message.reply(f"Ошибка при обращении к HF API: {e}")
        return
    await message.reply(answer)

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    await message.reply("Распознаю и решаю задачу с фото...")
    photo = message.photo[-1]
    file_path = f"temp_{message.message_id}.jpg"
    await photo.download(file_path)

    try:
        text = extract_text_from_image(file_path)
        if not text:
            await message.reply("Не удалось распознать текст на изображении.")
            return
        answer = await ask_model(text)
    except Exception as e:
        await message.reply(f"Ошибка: {e}")
        return
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

    await message.reply(answer)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
