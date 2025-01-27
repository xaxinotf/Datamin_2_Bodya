# app.py

import pandas as pd
import numpy as np
import os

# Для рекомендацій (Surprise)
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Dash і Plotly
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px

#################################
# 1. ЗАВАНТАЖЕННЯ ТА ОБРОБКА ДАНИХ
#################################

DATA_PATH = "../../OneDrive/Рабочий стол/ratings_Electronics (1).csv"
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Файл {DATA_PATH} не знайдено.")

# Читаємо CSV БЕЗ заголовка та присвоюємо імена стовпчикам
df_raw = pd.read_csv(
    DATA_PATH,
    header=None,
    names=['userId', 'productId', 'rating', 'timestamp']
)

# Вибірка до 1,000,000 рядків
if len(df_raw) > 1000000:
    df = df_raw.sample(n=1000000, random_state=42)
else:
    df = df_raw.copy()

# Перетворимо "timestamp" на дату (Unix time → datetime)
df['date'] = pd.to_datetime(df['timestamp'], unit='s')
df['year_month'] = df['date'].dt.to_period('M').astype(str)  # рік-місяць у рядковому форматі
df.drop(columns='timestamp', inplace=True)

#################################
# 2. EDA: ПЕРШИЙ ПОГЛЯД
#################################
unique_users = df['userId'].nunique()
unique_items = df['productId'].nunique()
print(f"Вибірка: {len(df)} рядків")
print(f"Унікальних користувачів: {unique_users}")
print(f"Унікальних товарів: {unique_items}")

mean_rating = df['rating'].mean()
print(f"Середній рейтинг: {mean_rating:.2f}")

# Додатковий EDA: Розподіл рейтингів
rating_counts = df['rating'].value_counts().sort_index()
rating_fig = px.bar(
    rating_counts,
    x=rating_counts.index,
    y=rating_counts.values,
    labels={'x': 'Rating', 'y': 'Count'},
    title='Розподіл Рейтингів',
    template='plotly_white'
)

# Топ-10 користувачів за кількістю оцінок
top_users = df['userId'].value_counts().nlargest(10)
top_users_fig = px.bar(
    top_users,
    x=top_users.values,
    y=top_users.index,
    orientation='h',
    labels={'x': 'Number of Ratings', 'y': 'UserId'},
    title='Топ-10 Користувачів за Кількістю Оцінок',
    template='plotly_white'
)

# Додатковий EDA: Розподіл кількості оцінок на товар
top_products = df['productId'].value_counts().nlargest(10)
top_products_fig = px.bar(
    top_products,
    x=top_products.values,
    y=top_products.index,
    orientation='h',
    labels={'x': 'Number of Ratings', 'y': 'ProductId'},
    title='Топ-10 Товарів за Кількістю Оцінок',
    template='plotly_white'
)

# Додатковий EDA: Тренд оцінок по місяцях
ratings_over_time = df.groupby('year_month').agg(
    total_ratings=('rating', 'count'),
    average_rating=('rating', 'mean')
).reset_index()

ratings_over_time_fig = px.line(
    ratings_over_time,
    x='year_month',
    y='total_ratings',
    title='Загальна Кількість Оцінок по Місяцях',
    labels={'year_month': 'Year-Month', 'total_ratings': 'Total Ratings'},
    template='plotly_white'
)

average_rating_over_time_fig = px.line(
    ratings_over_time,
    x='year_month',
    y='average_rating',
    title='Середній Рейтинг по Місяцях',
    labels={'year_month': 'Year-Month', 'average_rating': 'Average Rating'},
    template='plotly_white'
)

#################################
# 3. СТВОРЕННЯ РЕКОМЕНДАЦІЙНОЇ СИСТЕМИ
#################################
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

algo = SVD(n_factors=50, random_state=42)
algo.fit(trainset)

predictions = algo.test(testset)
rmse = accuracy.rmse(predictions, verbose=True)
mae = accuracy.mae(predictions, verbose=True)


#################################
# 4. ГЕНЕРАЦІЯ РЕКОМЕНДАЦІЙ
#################################
def get_top_n_recommendations(algo, user_id, df_full, n=5):
    """
    Для заданого user_id повернути top-n рекомендацій (продуктів),
    які користувач ще не оцінив.
    """
    # Знайдемо всі товари, які користувач НЕ оцінив
    user_data = df_full[df_full['userId'] == user_id]
    rated_items = set(user_data['productId'].unique())
    all_items = set(df_full['productId'].unique())
    unrated_items = list(all_items - rated_items)

    # Робимо передбачення рейтингу
    predictions = []
    for item_id in unrated_items:
        pred = algo.predict(str(user_id), str(item_id))
        predictions.append((item_id, pred.est))

    # Сортуємо за очікуваним рейтингом (спадно)
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:n]


# Приклад: Рекомендації для випадкового користувача
random_user = df['userId'].sample(1, random_state=42).iloc[0]
top_n = get_top_n_recommendations(algo, random_user, df, n=5)
print(f"Рекомендації для користувача {random_user}: {top_n}")


#################################
# 5. Time Series Line Chart
#################################
def generate_time_series_fig(df):
    """
    Генерує Plotly фігуру для Time Series Line Chart.
    Відображає загальну кількість оцінок та середній рейтинг по місяцях.
    """
    ratings_over_time = df.groupby('year_month').agg(
        total_ratings=('rating', 'count'),
        average_rating=('rating', 'mean')
    ).reset_index()

    fig = px.line(
        ratings_over_time,
        x='year_month',
        y=['total_ratings', 'average_rating'],
        title='Тренд Оцінок по Місяцях',
        labels={'year_month': 'Year-Month'},
        template='plotly_white',
        markers=True
    )

    fig.update_layout(
        xaxis_title="Year-Month",
        yaxis_title="Metrics",
        legend_title="Metrics",
        hovermode="x unified"
    )

    return fig


time_series_fig = generate_time_series_fig(df)

#################################
# 6. DASH-ДОДАТОК
#################################

app = dash.Dash(__name__)

# Precompute unique userIds for suggestions
unique_user_ids = df['userId'].unique()
unique_user_ids = unique_user_ids.tolist()

app.layout = html.Div([
    html.H1("Recommender System Dashboard", style={'textAlign': 'center'}),

    # Розділ з основними метриками
    html.Div([
        html.H2("Основні Метрики Моделі"),
        html.Div([
            html.P(f"RMSE на тесті: {rmse:.4f}"),
            html.P(f"MAE на тесті: {mae:.4f}"),
            html.P(f"Кількість користувачів (у вибірці): {unique_users}"),
            html.P(f"Кількість товарів (у вибірці): {unique_items}"),
            html.P(f"Середній рейтинг: {mean_rating:.2f}")
        ], style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'borderRadius': '5px'})
    ], style={'margin': '20px'}),

    # Розділ з додатковими візуалізаціями
    html.Div([
        html.H2("Додаткові Візуалізації"),
        dcc.Graph(id='rating-distribution', figure=rating_fig),
        dcc.Graph(id='top-users', figure=top_users_fig),
        dcc.Graph(id='top-products', figure=top_products_fig)
    ], style={'margin': '20px'}),

    # Розділ з Time Series Line Chart
    html.Div([
        html.H2("Тренд Оцінок по Місяцях"),
        dcc.Graph(id='time-series-chart', figure=time_series_fig)
    ], style={'margin': '20px'}),

    # Розділ з рекомендаціями
    html.Div([
        html.H2("Отримати Рекомендації"),
        html.Div([
            html.P("Введіть UserId для отримання топ-5 рекомендацій:"),
            dcc.Input(
                id='input-user',
                type='text',
                placeholder='Введіть UserId...',
                value=str(random_user),
                style={'width': '300px', 'padding': '5px'}
            ),
            dcc.Dropdown(
                id='user-suggestions',
                options=[],
                placeholder='Виберіть UserId',
                style={'width': '300px', 'marginTop': '5px', 'display': 'none'}
            ),
            html.Button(
                "Отримати Рекомендації",
                id='recommend-button',
                n_clicks=0,
                style={'marginTop': '10px', 'padding': '5px 10px'}
            )
        ], style={'marginBottom': '20px'}),
        html.Div(
            id='recommendation-result',
            style={'backgroundColor': '#f1f1f1', 'padding': '10px', 'borderRadius': '5px'}
        )
    ], style={'margin': '20px'}),

    # Додаткові Інструкції
    html.Div([
        html.H2("Інструкції"),
        html.Ul([
            html.Li("Перегляньте основні метрики моделі в розділі 'Основні Метрики Моделі'."),
            html.Li("Вивчайте додаткові візуалізації для розуміння поведінки користувачів та товарів."),
            html.Li("Аналізуйте тренди оцінок по місяцях у розділі 'Тренд Оцінок по Місяцях'."),
            html.Li("Введіть UserId у поле нижче (більше 3 символів) та виберіть з випадаючого списку для зручності."),
            html.Li("Натисніть кнопку 'Отримати Рекомендації', щоб отримати топ-5 рекомендацій для цього користувача.")
        ])
    ], style={'margin': '20px', 'backgroundColor': '#e7f3fe', 'padding': '10px', 'borderRadius': '5px'})
])


# CALLBACK для оновлення Time Series Line Chart при зміни даних (опціонально)
# Якщо ви хочете динамічно оновлювати графік при зміні метрик або інших параметрів, додайте відповідні callback.

# CALLBACK для оновлення списку користувачів (suggestions)
@app.callback(
    [Output('user-suggestions', 'options'),
     Output('user-suggestions', 'style')],
    Input('input-user', 'value')
)
def update_user_suggestions(input_value):
    if len(input_value) >= 3:
        # Фільтруємо userIds, що містять введений текст (case-insensitive)
        matching_user_ids = [uid for uid in unique_user_ids if input_value.lower() in uid.lower()]
        # Обмежимо кількість пропозицій, наприклад, до 10
        matching_user_ids = matching_user_ids[:10]
        options = [{'label': uid, 'value': uid} for uid in matching_user_ids]
        if options:
            return options, {'width': '300px', 'marginTop': '5px', 'display': 'block'}
    # Якщо менше 3 символів або нема пропозицій
    return [], {'width': '300px', 'marginTop': '5px', 'display': 'none'}


# CALLBACK для встановлення Input значення при виборі з Dropdown
@app.callback(
    Output('input-user', 'value'),
    Input('user-suggestions', 'value'),
    State('input-user', 'value')
)
def set_input_value(selected_user, current_input):
    if selected_user:
        return selected_user
    return current_input


# CALLBACK для генерації рекомендацій
@app.callback(
    Output('recommendation-result', 'children'),
    Input('recommend-button', 'n_clicks'),
    State('input-user', 'value')
)
def update_recommendations(n_clicks, user_id_value):
    if n_clicks > 0:
        try:
            user_id_value = str(user_id_value)
            if user_id_value not in df['userId'].unique():
                return html.P(f"Користувач {user_id_value} не знайдений.", style={'color': 'red'})
            top_n_recs = get_top_n_recommendations(algo, user_id_value, df, n=5)
            if len(top_n_recs) == 0:
                return f"Користувач {user_id_value} уже оцінив усі товари або не знайдено даних."
            else:
                # Можна додати більше деталей про товари, якщо є додаткові дані
                return [
                    html.P(f"Рекомендації для користувача {user_id_value}:"),
                    html.Ul([html.Li(f"Товар: {prod}, очікуваний рейтинг: {est:.2f}") for prod, est in top_n_recs])
                ]
        except Exception as e:
            return f"Помилка при обчисленні рекомендацій: {e}"
    return ""


if __name__ == '__main__':
    app.run_server(debug=True)
