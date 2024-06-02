from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Символы для распознавания
symbols = ['A', 'B', 'C', 'D']

# Пути к шрифтам
train_fonts = ['arial.ttf', 'ariblk.ttf', 'calibri.ttf', 'Neo Sans.ttf']
test_font = 'ONYX.ttf'

# Размер изображения и шрифта
image_size = (16, 16)
font_size = 14

# Директории для сохранения изображений
train_dir = 'train_images'
test_dir = 'test_images'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def generate_image(symbol, font_path, save_path):
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new('L', image_size, color=255)  # белый фон
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), symbol, font=font)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((image_size[0]-width)/2, (image_size[1]-height)/2), symbol, font=font, fill=0)  # черный символ
    image.save(save_path)

# Генерация обучающей выборки
for i, symbol in enumerate(symbols):
    for j, font in enumerate(train_fonts):
        save_path = os.path.join(train_dir, f'{symbol}_{j}.png')
        generate_image(symbol, font, save_path)

# Генерация тестовой выборки
for i, symbol in enumerate(symbols):
    save_path = os.path.join(test_dir, f'{symbol}.png')
    generate_image(symbol, test_font, save_path)

print("Данные сгенерированы и сохранены.")
