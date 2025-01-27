# Система Рекомендацій для Електроніки

Цей проект представляє собою веб-додаток для системи рекомендацій, розроблений за допомогою Python, Dash та бібліотеки Surprise (`scikit-surprise`). Система аналізує відгуки користувачів про електронні товари та надає персоналізовані рекомендації на основі їхніх уподобань.

## Технології

- **Python 3.7+**
- **Pandas** - для обробки та аналізу даних
- **NumPy** - для чисельних операцій
- **Scikit-learn** - для машинного навчання та обчислення схожості
- **Surprise (scikit-surprise)** - для побудови та аналізу систем рекомендацій
- **Dash** - фреймворк для створення веб-додатків
- **Plotly** - для створення інтерактивних графіків

## Вимоги

Перед початком роботи переконайтеся, що у вас встановлено наступне:

- **Python 3.7 або вище**
- **pip** - пакетний менеджер для Python
- **Git** - для клонування репозиторію (опціонально)

## Інсталяція




### 1. Створення Віртуального Середовища

Рекомендується використовувати віртуальне середовище для ізоляції залежностей проєкту.

#### За допомогою `venv`:

```bash
python -m venv venv
```

#### Активація Віртуального Середовища:

- **На Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **На macOS та Linux:**

  ```bash
  source venv/bin/activate
  ```

### 2. Встановлення Залежностей

Встановіть необхідні бібліотеки за допомогою `pip`:

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn scikit-surprise dash plotly
```

**Примітка:** Якщо у вас виникають проблеми з встановленням `scikit-surprise`, спробуйте встановити його через `conda` (якщо використовуєте Anaconda):

```bash
conda install -c conda-forge scikit-surprise
```

або встановіть з вихідного коду:

```bash
pip install git+https://github.com/NicolasHug/Surprise.git
```

## Запуск Додатку

1. **Переконайтеся, що ваш CSV-файл має правильну структуру** та знаходиться за вказаним шляхом `"../../OneDrive/Рабочий стол/ratings_Electronics (1).csv"`. Структура файлу повинна бути наступною:

    ```
    userId,productId,rating,timestamp
    ```

2. **Запустіть скрипт** `app.py` у вашому середовищі розробки або з командного рядка:

    ```bash
    python app.py
    ```

3. **Відкрийте веб-браузер** та перейдіть за адресою, яку виведе Dash (зазвичай [http://127.0.0.1:8050/](http://127.0.0.1:8050/)).

## Використання

Після запуску додатку ви побачите інтерфейс з наступними секціями:

### 1. Основні Метрики Моделі

Відображає ключові метрики моделі:

- **RMSE на тесті:** Корінь середньоквадратичної помилки.
- **MAE на тесті:** Середня абсолютна помилка.
- **Кількість користувачів:** Унікальні користувачі у вибірці.
- **Кількість товарів:** Унікальні товари у вибірці.
- **Середній рейтинг:** Середнє значення оцінок.

### 2. Додаткові Візуалізації

Включає три інтерактивні графіки:

- **Розподіл Рейтингів:** Гістограма, що показує, які рейтинги найбільш поширені.
- **Топ-10 Користувачів за Кількістю Оцінок:** Горизонтальний бар-чарт, що демонструє найактивніших користувачів.
- **Топ-10 Товарів за Кількістю Оцінок:** Горизонтальний бар-чарт, що демонструє найпопулярніші товари.

### 3. Тренд Оцінок по Місяцях

Лінійний графік, що відображає:

- **Загальна Кількість Оцінок по Місяцях**
- **Середній Рейтинг по Місяцях**

### 4. Отримати Рекомендації

Ця секція дозволяє отримувати персоналізовані рекомендації:

1. **Введіть `UserId`:** У текстове поле введіть ідентифікатор користувача (більше 3 символів).
2. **Виберіть `UserId` з Dropdown:** Якщо введено більше 3 символів, з'являться пропозиції для швидкого вибору.
3. **Натисніть "Отримати Рекомендації":** Кнопка для генерації топ-5 рекомендацій для обраного користувача.
4. **Відображення Рекомендацій:** Список рекомендованих товарів з очікуваними рейтингами.


## Додаткова Інформація

### Оптимізація Генерації Рекомендацій

Функція `get_top_n_recommendations` може бути повільною при роботі з великими наборами даних, оскільки робить передбачення для кожного неоціненого товару. Для покращення продуктивності можна використовувати:

- **Методи схожості:** Використання схожості між товарами або користувачами для швидшої генерації рекомендацій.
- **Кешування:** Збереження передбачених рейтингів для часто використовуваних запитів.





Recommender System


•	Objective: To build a recommender system that suggests products or movies to users based on their preferences and behaviors.
•	Key Concepts: Collaborative Filtering, User-Item Interactions, Machine Learning.

Tools and Technologies
•	Libraries:
•	pandas for data manipulation.
•	scikit-learn for basic machine learning tasks.
•	Surprise or scikit-surprise for building and analyzing recommender systems.
Dataset
•	Amazon Product Reviews

Tasks Breakdown

1. Data Preprocessing
•	Loading Data: Import the dataset and understand its structure.
•	Cleaning and Transformation: Handle missing values, normalize data, and transform it into a suitable format for building recommender systems.
2. Exploratory Data Analysis (EDA)
•	Analyze user behavior and item characteristics.
•	Visualize the distribution of ratings, popular items, active users, etc.
3. Building the Recommender System
•	Collaborative Filtering: Implement collaborative filtering methods, such as user-based, item-based, or matrix factorization techniques.
•	Model Training: Train the model using the dataset.
•	Tuning and Optimization: Fine-tune the model parameters for optimal performance.
4. Evaluation
•	Split the data into training and testing sets.
•	Evaluate the recommender system using metrics like RMSE (Root Mean Square Error), MAE (Mean Absolute Error), precision, recall, etc.
5. Making Recommendations
•	Generate recommendations for a set of users.
•	Analyze the quality of recommendations (e.g., diversity, novelty).
6. Visualization and Interpretation
•	Visualize user-item interactions.
•	Interpret the recommendation results and model behavior.
•	Interpret the recommendation results and model behavior.






Деталі та Пояснення Покращень
1. Data Preprocessing
Завантаження та Обробка Даних
Ми завантажуємо дані з файлу CSV, присвоюємо імена стовпчикам та перетворюємо Unix timestamp на читабельний формат дати. Це необхідно для подальшого аналізу та візуалізації.
Обробка Вибірки
Якщо ваш датасет занадто великий, ви можете обмежити кількість рядків для обробки, використовуючи метод sample. Проте в цьому випадку ми обробляємо всі дані.
2. Exploratory Data Analysis (EDA)
Основні Метрики
Виводимо кількість рядків, унікальних користувачів та товарів, а також середній рейтинг. Це допомагає зрозуміти розмір та якість датасету.
Візуалізація Розподілу Рейтингів
Гістограма розподілу рейтингів дозволяє побачити, які рейтинги найбільш поширені серед користувачів.
Топ-10 Користувачів за Кількістю Оцінок
Горизонтальний бар-чарт демонструє найактивніших користувачів, що може бути корисно для розуміння поведінки користувачів.
3. Building the Recommender System
Використання Алгоритму SVD
Алгоритм SVD з бібліотеки Surprise використовується для створення матриці користувач-товар та прогнозування рейтингів.
4. Evaluation
Метрики Оцінки Моделі
Використовуємо RMSE (Root Mean Square Error) та MAE (Mean Absolute Error) для оцінки точності моделі. Ці метрики показують, наскільки прогнозовані рейтинги відрізняються від реальних.
5. Making Recommendations
Генерація Топ-5 Рекомендацій
Функція get_top_n_recommendations генерує рекомендації для заданого користувача, пропонуючи товари, які користувач ще не оцінив, з найвищими прогнозованими рейтингами.
6. Visualization and Interpretation
Animated Bubble Chart
Анімований Bubble Chart дозволяє відображати всі доступні дані, використовуючи розмір та колір бульбашок для представлення різних метрик. Це забезпечує більш гнучке та інформативне відображення даних порівняно з Bar Chart Race.
•	Розмір бульбашки: Відображає кількість оцінок (популярність).
•	Колір бульбашки: Відображає середній рейтинг.
•	Анімація: Показує зміну метрик протягом часу (по місяцях).
Інтерактивний Інтерфейс
Користувачі можуть вибирати метрику для відображення на Bubble Chart, вводити UserId для отримання рекомендацій та переглядати додаткові візуалізації для глибшого аналізу даних.

