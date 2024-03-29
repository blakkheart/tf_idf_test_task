import os

import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import redis
import concurrent.futures
from functools import partial, reduce
from itertools import islice

import time

load_dotenv()

HOST = os.getenv('HOST', 'localhost')
PORT = os.getenv('PORT', 6379)

db = redis.Redis(host='localhost', port=6379, decode_responses=True)
app = FastAPI()

templates = Jinja2Templates(directory='src/templates')


def partition(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def map_freqeuncies(chunk=60000):
    counter = {}
    counter['words_count'] = 0
    for line in chunk:
        words = line.decode('Windows-1251')
        words = words.translate(str.maketrans('', '', '!@#`.;:?\'"$,)(][')).split()
        for word in words:
            if word.isdigit():
                continue
            word = word.lower()
            if counter.get(word):
                counter[word] += 1
            else:
                counter[word] = 1
            counter['words_count'] += 1
    return counter


def merge_dic(first, second):
    for key in second:
        if key in first:
            first[key] += second[key]
        else:
            first[key] = second[key]
    return first


def dict_chunks(data, size=1000):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def set_resulst(data, words_count, all_documents_count):
    result = []
    for word, word_count in data.items():
        count_word_per_doc = db.hincrby('count_word_per_doc', word, 1)
        result.append(
            {
                'word': word,
                'tf': round(word_count / words_count, 5),
                'idf': round(int(all_documents_count) / int(count_word_per_doc), 5),
            }
        )
    return result


@app.post('/file')
async def upload_file(file: UploadFile = File(...)):
    time_start = time.time()

    partition_size = 60000

    with file.file as f:
        contents = f.readlines()
        loop = asyncio.get_running_loop()
        tasks = []

        with concurrent.futures.ProcessPoolExecutor() as pool:
            for chunk in partition(contents, partition_size):
                tasks.append(loop.run_in_executor(pool, partial(map_freqeuncies, chunk)))

        task_result = await asyncio.gather(*tasks)
        final_result = reduce(merge_dic, task_result)

    time_middle = time.time()
    print(time_middle - time_start)

    all_documents_count = db.hincrby('all_documents_count', 'count', 1)
    words_count = final_result.get('words_count')
    size = 1000
    tasks = []
    final_result.pop('words_count')

    with concurrent.futures.ThreadPoolExecutor() as thread:
        for chunk in dict_chunks(final_result, size):
            tasks.append(
                loop.run_in_executor(
                    thread,
                    partial(set_resulst, chunk, words_count, all_documents_count),
                )
            )
    task_result = await asyncio.gather(*tasks)
    final_result = reduce(lambda x, y: x + y, task_result)

    time_end = time.time()
    print(time_end - time_middle)
    print(time_end - time_start)

    return sorted(final_result, key=lambda x: -x['idf'])


# @app.post('/file')
# async def upload_file(file: UploadFile = File(...)):
#     time_start = time.time()
#     tf_words = {}
#     words_count = 0

#     for line in file.file.readlines():
#         words = line.strip().split()
#         for word in words:
#             word = word.decode('Windows-1251').lower().strip(',.')
#             if tf_words.get(word):
#                 tf_words[word] += 1
#             else:
#                 tf_words[word] = 1
#             words_count += 1
#     time_middle = time.time()
#     print(time_middle - time_start)
#     all_documents_count = db.hincrby('all_documents_count', 'count', 1)
#     result = []

#     for word, word_count in tf_words.items():
#         count_word_per_doc = db.hincrby('count_word_per_doc', word, 1)
#         result.append(
#             {
#                 'word': word,
#                 'tf': round(word_count / words_count, 5),
#                 'idf': round(int(all_documents_count) / int(count_word_per_doc), 5),
#             }
#         )
#     time_end = time.time()
#     print(time_end - time_middle)
#     print(time_end - time_start)
#     return sorted(result, key=lambda x: -x['idf'])


@app.get('/', response_class=HTMLResponse, include_in_schema=False)
async def main_page(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')
