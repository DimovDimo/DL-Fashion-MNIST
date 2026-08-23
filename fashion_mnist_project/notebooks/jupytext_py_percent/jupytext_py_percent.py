# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jupytext_py_percent//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# !pip install -q jupytext --upgrade

# %%
import os

# 1. Името на папката, която ще се създаде автоматично в същата директория
output_dir = "jupytext_py_percent"
os.makedirs(output_dir, exist_ok=True)

# 2. Указваме на Jupytext да записва .py файловете в подпапката jupytext_py_percent
# Използваме директно *.ipynb, тъй като вече сме в правилната папка
# !jupytext --set-formats "ipynb,jupytext_py_percent//py:percent" *.ipynb

# 3. Стартиране на двупосочната синхронизация
# !jupytext --sync *.ipynb

print(f" Готово! Всички .py файлове са генерирани в подпапка: {output_dir}")


# %%
