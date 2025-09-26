import os
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from dotenv import load_dotenv

load_dotenv()  # загружаем переменные из .env
TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# Массив слов-триггеров
TRIGGERS = ["тахохов", "гоша", "гоши", "гошан", "гашан", "георгий"]

# Стикер
STICKER_ID = "CAACAgIAAxkBAAEPck5o1mg46qdyJBIQ5VRWGa63LF_SoAAC9YAAAmsuuUocW-dcBOCJVDYE"

# Ответ в личке
PRIVATE_REPLY = "не пиши мне"

@dp.message_handler(content_types=types.ContentType.TEXT)
async def check_messages(message: types.Message):
    # Проверяем: если чат приватный (личка)
    if message.chat.type == "private":
        await message.answer(PRIVATE_REPLY)
        return

    # Если это группа — проверяем триггеры
    text = message.text.lower()
    if any(word in text for word in TRIGGERS):
        await message.reply_sticker(STICKER_ID)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
