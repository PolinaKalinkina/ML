import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from joblib import dump, load
import os
import json

class ArticleClassifier:
    def __init__(self):
        self.pipeline = None

        self.cluster_names = {
            0: "Регистрация и настройка профиля",
            1: "Работа с каталогом (СТЕ, котировки)",
            2: "Виды закупок",
            3: "Подключение контракта и документооборот"
        }


        self.manual_mapping = {

            "Новая версия опубликованного шаблона": 0,
            "Утверждение заявки на регистрацию новой организации": 0,
            "Разблокировка пользователя": 0,
            "Как заполнить раздел «Статистические коды» при подаче заявки на регистрацию компании?": 0,
            "При подаче заявки на регистрацию компании возникает ошибка «Не заполнены поля «Банковский идентификационный код» и «Расчетный счет»": 0,


            "Как добавить характеристики, если их нет в выпадающем списке": 1,
            "Отображение оферты": 1,
            "Удаление оферты": 1,
            "Статус оферты 'Ввод сведений'": 1,
            "Архивирование оферты": 1,


            "Уведомления о наступающих сроках исполнения этапа": 2,


            "Заявка в статусе 'Черновик', сколько ждать рассмотрения заявки?": 3,
            "Как на Портале поставщиков пользователю ознакомиться с карточкой поставщика/ карточкой заказчика своей организации?": 3,
            "Как на Портале поставщиков ознакомиться с информацией по регионам?": 3,
            "Заявка в статусе 'Редактирование', сколько ждать рассмотрения заявки": 3,
            "Ошибка при подписании оферты (Ошибка инициализации объекта хэша)": 3
        }

    def load_data(self, filepath):
        """Загрузка данных"""
        try:
            if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                return pd.read_excel(filepath)
            elif filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return None

    def preprocess(self, text):
        """Базовая очистка текста"""
        if pd.isna(text):
            return ""
        return str(text).lower().strip()

    def add_manual_labels(self, data, title_col='Заголовок статьи'):
        """Добавление ручных меток на основе заголовков"""
        data['cluster'] = data[title_col].map(self.manual_mapping)
        return data

    def train_model(self, data, text_col='Заголовок статьи'):
        """Обучение модели"""
        vectorizer = TfidfVectorizer(
            stop_words=["как", "при", "о", "на", "с"],
            max_features=500
        )

        kmeans = KMeans(n_clusters=4, random_state=42)

        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('clustering', kmeans)
        ])

        labeled_data = data.dropna(subset=['cluster'])
        if len(labeled_data) >= 4:
            X = labeled_data[text_col].apply(self.preprocess)
            y = labeled_data['cluster']
            self.pipeline.fit(X, y)
        else:
            print("Недостаточно размеченных данных для обучения")

    def predict(self, new_texts):
        """Предсказание для новых текстов"""
        if self.pipeline is None:
            raise ValueError("Модель не обучена")

        if isinstance(new_texts, str):
            new_texts = [new_texts]

        clusters = self.pipeline.predict(new_texts)

        results = []
        for text, cluster in zip(new_texts, clusters):
            results.append({
                'text': text,
                'cluster_id': int(cluster),
                'cluster_name': self.cluster_names.get(cluster, "Unknown")
            })

        return results

    def save_model(self, dir_path='model'):
        """Сохранение модели и метаданных"""
        os.makedirs(dir_path, exist_ok=True)
        dump(self.pipeline, os.path.join(dir_path, 'pipeline.joblib'))

        metadata = {
            'cluster_names': self.cluster_names,
            'manual_mapping': self.manual_mapping
        }
        with open(os.path.join(dir_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def load_model(self, dir_path='model'):
        """Загрузка модели"""
        self.pipeline = load(os.path.join(dir_path, 'pipeline.joblib'))

        with open(os.path.join(dir_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            self.cluster_names = metadata['cluster_names']
            self.manual_mapping = metadata['manual_mapping']


if __name__ == "__main__":
    classifier = ArticleClassifier()

    data = classifier.load_data("/content/Статьи.xls")
    if data is not None:
        data = classifier.add_manual_labels(data)
        classifier.train_model(data)


        new_articles = [
            "Как зарегистрироваться?",
            "Как добавить характеристики, если их нет в выпадающем списке",
            "Уведомления о наступающих сроках исполнения этапа",
            "Ошибка при подписании оферты (Ошибка инициализации объекта хэша)"
        ]

        predictions = classifier.predict(new_articles)
        for pred in predictions:
            print(f"Заголовок: '{pred['text']}'")
            print(f"Категория: {pred['cluster_id']} - {pred['cluster_name']}\n")

        classifier.save_model()
