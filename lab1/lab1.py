import numpy as np

# Параметры
input_size = 256  # предположим, что каждый образ представлен 16x16 пикселями
output_size = 4   # 4 символа

# Инициализация весов
weights = np.random.uniform(-0.03, 0.03, (output_size, input_size + 1))

print("Начальные веса и смещения:")
print(weights)

from PIL import Image
import os

# Функция для преобразования изображения в вектор
def image_to_vector(image_path):
    image = Image.open(image_path).convert('L')  # преобразуем в оттенки серого
    image = image.resize((16, 16))  # изменяем размер до 16x16
    vector = np.array(image).flatten()  # преобразуем изображение в вектор
    vector = vector / 255  # нормализуем значения
    return vector

# Подготовка данных
def load_data(data_dir, symbols):
    X = []
    D = []
    for i, symbol in enumerate(symbols):
        symbol_dir = os.path.join(data_dir, symbol)
        for file_name in os.listdir(symbol_dir):
            file_path = os.path.join(symbol_dir, file_name)
            vector = image_to_vector(file_path)
            X.append(vector)
            d = np.zeros(len(symbols))
            d[i] = 1
            D.append(d)
    return np.array(X), np.array(D)

# Загрузка данных
train_dir = 'train_images'
test_dir = 'test_images'

symbols = ['A', 'B', 'C', 'D']

X_train, D_train = load_data(train_dir, symbols)
X_test, D_test = load_data(test_dir, symbols)

print("Данные загружены.")
print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

# Параметры
learning_rate = 0.1
epochs = 100

# Инициализация весов
weights = np.random.uniform(-0.03, 0.03, (output_size, input_size + 1))

# Функция активации
def activation_function(x):
    return 1 if x >= 0 else 0

# Обучение персептрона
def train_perceptron(X, D):
    global weights
    for epoch in range(epochs):
        for x, d in zip(X, D):
            x = np.append(1, x)  # добавляем единицу для учета смещения
            y = np.dot(weights, x)
            y = np.array([activation_function(i) for i in y])
            error = d - y
            weights += learning_rate * np.outer(error, x)

# Предсказание
def predict(x):
    x = np.append(1, x)
    y = np.dot(weights, x)
    return np.array([activation_function(i) for i in y])

# Обучение
train_perceptron(X_train, D_train)

# Проверка на тестовой выборке
predictions = [predict(x) for x in X_test]
accuracy = np.mean([np.array_equal(p, d) for p, d in zip(predictions, D_test)])

print("Точность на тестовой выборке:", accuracy)
