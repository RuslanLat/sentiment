from datetime import date
from pony.orm import *
import json

with open('DATABASE_URL.txt', 'r', encoding='utf-8') as f:
    DATABASE_URL = json.load(f)

db = Database()

# подключение к базе данных 'sentiment' PostgreSQL
db.bind(**DATABASE_URL)

class Sentiment(db.Entity):
    """комментарии (отзывы) пользователей"""
    _resource_ = 'https://www.imdb.com/'
    sentiment_id = PrimaryKey(int, column='sentiment_id', auto=True)  # идентификатор комментария (отзыва)
    rating = Optional(int, column='rating')  # рейтинг комментария (отзыва)
    review = Required(str, column='review')  # комментарий (отзыв) пользователя
    label_id = Required('Label')  # идентификатор типа комментария (отзыва)
    data_id = Required('Data', column='data_id')  # идентификатор выборки
    title_id = Required('Title', column='title_id')  # идентификатор url фильма


class Title(db.Entity):
    """url фильмов"""
    _resource_ = 'https://www.imdb.com/'
    title_id = Set(Sentiment)  # идентификатор url фильма
    title_url = Required(str, unique=True, column='title_url')  # url фильма


class Data(db.Entity):
    """тип выборки"""
    _resource_ = 'https://www.imdb.com/'
    data_id = Set(Sentiment)  # идентификатор выборки
    data_name = Required(str, unique=True, column='data_name')  # наименование выборки


class Label(db.Entity):
    """тип комментариев (отзывов)"""
    _resource_ = 'https://www.imdb.com/'
    label_id = Set(Sentiment)  # идентификатор типа комментария (отзыва)
    label_name = Required(str, unique=True, column='label_name')  # тип комментария (отзыва)


class Sentiment_New(db.Entity):
    """комментарии (отзывы) пользователей"""
    sentiment_id = PrimaryKey(int, column='sentiment_id', auto=True)  # идентификатор комментария (отзыва)
    date = Required(date, column='date')  # дата комментария (отзыва)
    review = Required(str, column='review')  # комментарий (отзыв) пользователя
    rating_pred = Optional(int, column='rating_pred')  # предсказанный рейтинг комментария (отзыва)

db.generate_mapping()