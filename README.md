

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

