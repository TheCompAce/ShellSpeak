import os
import sqlite3
import json

class ResponseCache:
    def __init__(self, cache_file=None):
        self.cache_file = os.path.abspath(cache_file) if cache_file else None
        self.cache = {}  # In-memory cache
        if cache_file:
            self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS cache
                            (system_prompt TEXT, user_prompt TEXT, response TEXT, PRIMARY KEY(system_prompt, user_prompt))''')

    def get(self, system_prompt, user_prompt):
        if self.cache_file:
            print(self.cache_file)
            with sqlite3.connect(self.cache_file) as conn:
                cur = conn.execute('SELECT response FROM cache WHERE system_prompt=? AND user_prompt=?', (system_prompt, user_prompt))
                row = cur.fetchone()
                return row[0] if row else None
        return self.cache.get((system_prompt, user_prompt))

    def set(self, system_prompt, user_prompt, response):
        if self.cache_file:
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('INSERT OR REPLACE INTO cache VALUES (?, ?, ?)', (system_prompt, user_prompt, response))
        else:
            self.cache[(system_prompt, user_prompt)] = response

