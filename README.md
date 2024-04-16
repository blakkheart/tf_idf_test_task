
# 🔍 TF и IDF 🔍

## Описание

Веб-приложение. В качестве интерфейса - страница с формой для загрузки текстового файла (принимает в себя только ```.txt```), после загрузки и обработки файла отображается таблица с 50 словами с колонками:

-   ***слово***.
-   ***tf***, сколько раз это слово встречается в тексте.
-   ***idf***, обратная частота документа.

Вывод упорядочен по уменьшению idf и по уменьшению tf.

Для примера: обработка 1 книги Войны и Мир занимает в районе 7 секунд, после чего пользователю предоставляется информация.

## Стэк технологий

-  [FastAPI](https://fastapi.tiangolo.com/)  — фреймворк.
-  [Redis](https://redis.io/) — база данных приложения.
- [Docker](https://www.docker.com/) — контейнеризация приложения.
- [Jinja2](https://pypi.org/project/Jinja2/)  +  [Bootstrap](https://getbootstrap.com/)  — для небольшой визуальной части проекта.

## Установка

1. Склонируйте репозиторий:
```bash
git clone https://github.com/blakkheart/tf_idf_test_task.git
```
2. Перейдите в директорию проекта:
```bash
cd tf_idf_test_task
```
3. Установите и активируйте виртуальное окружение:
   - Windows
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   - Linux/macOS
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Обновите [pip](https://pip.pypa.io/en/stable/):
   - Windows
   ```bash
   (venv) python -m pip install --upgrade pip
   ```
   - Linux/macOS
   ```bash
   (venv) python3 -m pip install --upgrade pip
   ```
5. Установите зависимости из файла requirements.txt:
   ```bash
   (venv) pip install -r requirements.txt
   ```
Создайте и заполните файл `.env` по примеру с файлом `.env.example`, который находится в корневой директории.



## Использование  

1. Введите команду для запуска докер-контейнера:
	```bash
	docker compose up
	```

Сервер запустится по адресу ```127.0.0.1:8000```
Вы можете посмотреть документацию по адресу ```127.0.0.1:8000/docs```

