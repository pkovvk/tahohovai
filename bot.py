import os
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from huggingface_hub import InferenceClient

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
MODEL = os.getenv("MODEL")  # deepseek-ai/DeepSeek-V3.1-Terminus:novita

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))

def extract_text_from_image(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img, lang='eng+rus')
    return text

async def ask_model(prompt, image_path=None):
    image_bytes = None
    if image_path:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    response = hf_client.generate(
        model=MODEL,
        inputs=prompt,
        image=image_bytes
    )
    return response[0]['generated_text']

@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    await message.reply("Обрабатываю задачу...")
    answer = await ask_model(message.text)
    await message.reply(answer)

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    await message.reply("Распознаю и решаю задачу с фото...")
    photo = message.photo[-1]
    file_path = f"temp_{message.message_id}.jpg"
    await photo.download(file_path)
    
    text = extract_text_from_image(file_path)
    answer = await ask_model(text, image_path=file_path)
    
    os.remove(file_path)
    await message.reply(answer)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
