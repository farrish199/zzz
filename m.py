from pyrogram import Client, filters
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Muat turun model dan tokenizer GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Inisialisasi bot
app = Client("my_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

@app.on_message(filters.command("ask") & filters.text)
def ask_gpt(client, message):
    # Mengambil pertanyaan selepas /ask
    question = message.text[len("/ask "):]
    
    if question:
        try:
            # Tokenize input
            inputs = tokenizer.encode(question, return_tensors='pt')
            
            # Hasilkan jawapan
            with torch.no_grad():
                outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            answer = f"Terjadi ralat: {str(e)}"
        
        # Hantar jawapan kembali kepada pengguna
        message.reply_text(answer)
    else:
        message.reply_text("Sila taip soalan selepas /ask.")

# Mulakan bot
app.run()
