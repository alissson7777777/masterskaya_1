from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, select_autoescape
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from heart_model import HeartAttackModel, FEATURES

# иницализация модели 
model = HeartAttackModel()

#подставляем данные (например, вероятность инфаркта) в HTML прямо внутри кода, без отдельных шаблонных файлов.
_render = lambda tpl, **ctx: Environment(loader=None, autoescape=select_autoescape(['html'])).from_string(tpl).render(**ctx)

# Pydantic‑схема, которая ограничивает признаки и задает их типы, наследую Basemodel
#зададим ограничения, чтобы пользовательно не вводил значения меньше 0 для возраста, например
class Patient(BaseModel):
    age: int = Field(ge=0)
    cholesterol: float
    heart_rate: int = Field(ge=40, le=120, alias='heart rate')
    diabetes: Literal[0,1]
    smoking: Literal[0,1]
    obesity: Literal[0,1]
    alcohol_consumption: int = Field(ge=0, alias='alcohol consumption')
    exercise_hours_per_week: float = Field(ge=0, alias='exercise hours per week')
    stress_level: int = Field(ge=0, le=10, alias='stress level')
    sedentary_hours_per_day: float = Field(ge=0, alias='sedentary hours per day')
    bmi: float = Field(ge=10)
    physical_activity_days_per_week: int = Field(ge=0, alias='physical activity days per week')
    sleep_hours_per_day: float = Field(ge=0, alias='sleep hours per day')
    troponin: float = Field(ge=0)
    gender: Literal[0,1]
    systolic_blood_pressure: int = Field(ge=0, alias='systolic blood pressure')

    class Config:
        populate_by_name = True  




#иницализация fastapi
app = FastAPI(title='Ассистент кардиолога', docs_url='/api/docs', redoc_url=None)

FIELDS = [
    ('Возраст (лет)', 'age'),
    ('Холестерин (ммоль/л)', 'cholesterol'),
    ('Частота сердечных сокращений (уд/мин)', 'heart_rate'),
    ('Сахарный диабет (0/1)', 'diabetes'),
    ('Курение (0/1)', 'smoking'),
    ('Ожирение (0/1)', 'obesity'),
    ('Потребление алкоголя (порций/нед.)', 'alcohol_consumption'),
    ('Физическая активность (ч/нед.)', 'exercise_hours_per_week'),
    ('Уровень стресса (1–10)', 'stress_level'),
    ('Сидячие часы в день', 'sedentary_hours_per_day'),
    ('Индекс массы тела', 'bmi'),
    ('Дни активности в неделю', 'physical_activity_days_per_week'),
    ('Сон (ч/день)', 'sleep_hours_per_day'),
    ('Тропонин (нг/мл)', 'troponin'),
    ('Пол (0 – жен., 1 – муж.)', 'gender'),
    ('Систолическое давление (мм рт. ст.)', 'systolic_blood_pressure'),
]

#отрисовка frontenda, то есть html формы, которая используется внутри _render
HTML_PAGE = """<!DOCTYPE html><html lang='ru' class='h-full'>
<head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>Риск инфаркта</title><script src='https://cdn.tailwindcss.com'></script></head>
<body class='bg-gradient-to-br from-red-50 via-rose-50 to-indigo-50 min-h-full flex flex-col items-center py-10'>
<h1 class='text-3xl font-semibold mb-6'>Предсказание риска инфаркта</h1>
<form action='/web_predict' method='post' class='bg-white shadow-lg rounded-2xl p-8 w-full max-w-4xl grid grid-cols-1 sm:grid-cols-2 gap-6'>
{% for label, name in fields %}
  <div class='flex flex-col'>
    <label class='text-sm font-medium mb-1' for='{{ name }}'>{{ label }}</label>
    <input class='border rounded-lg px-3 py-2' name='{{ name }}' id='{{ name }}' required>
  </div>
{% endfor %}
  <div class='sm:col-span-2 text-center'>
    <button type='submit' class='bg-red-600 text-white px-6 py-3 rounded-xl hover:bg-red-700 transition'>Рассчитать</button>
  </div>
</form>
{% if error %}
  <p class='text-red-600 mt-4 font-medium'>{{ error }}</p>
{% endif %}
{% if level is not none %}
  <div class='mt-8 w-full max-w-xl text-center'>
    <div class='bg-white shadow-xl rounded-2xl p-6'>
      <h2 class='text-xl font-semibold mb-4'>Результат</h2>
      <p class='text-lg'>Уровень риска: <span class='font-bold {% if level == "high" %}text-red-600{% else %}text-green-600{% endif %}'>
        {{ 'Высокий' if level == 'high' else 'Низкий' }}</span>
        <span class='text-2xl ml-2'>{{ '🟥' if level == 'high' else '🟩' }}</span></p>
    </div>
  </div>
{% endif %}
</body></html>"""


#обработчик get-запроса по адресу по корнеовму адресу
@app.get('/', response_class=HTMLResponse, tags=["Предсказание рисков сердечного приступа"])
async def index(): 
    # возвращает HTML-страницу с формой для ввода данных пациента
    return HTMLResponse(_render(HTML_PAGE, fields=FIELDS, level=None, error=None))

#обрабатывает post-запрос с формы /web_predict и делает предсказание на http://127.0.0.1:8000
@app.post('/web_predict', response_class=HTMLResponse, tags=["Предсказание рисков сердечного приступа"])
async def web_predict(request: Request):
    #получаем все данные, введённые пользователем в HTML-форму в виде словаря field_name - значение
    form = await request.form()

    #получает данные
    # превращаем строки в числа, если возможно
    data: dict[str, object] = {}
    for k, v in form.items():
        s = v.strip().replace(',', '.')  # убираем лишние пробелы и заменяем запятую на точку если что
        try:
            num = float(s)  # пробуем преобразовать в число
            data[k] = int(num) if num.is_integer() else num  # если целое — делаем int
        except ValueError:
            data[k] = s  # если не число — оставляем строкой

    # валидация данных с помощью pydantic
    try:
        patient = Patient(**data)
    except ValidationError as exc:
        msg = exc.errors()[0].get('msg', 'Некорректные данные')
        return HTMLResponse(_render(HTML_PAGE, fields=FIELDS, level=None, error=msg), status_code=400)

    # полготовка списка значений в нужном порядке и делает предсказание, возвращает HTML с результатом или ошибкой
    X = pd.DataFrame([patient.model_dump(by_alias=True)])
    level = 'high' if model.predict(X)[0] else 'low'
    return HTMLResponse(_render(HTML_PAGE, fields=FIELDS, level=level, error=None))


#apiверсия предсказания, обработка POST-запроса на /api/predict с JSON-данными
@app.post('/api/predict', tags=["Предсказание рисков сердечного приступа"])
async def api_predict(patient: Patient):
    X = pd.DataFrame([patient.model_dump(by_alias=True)])
    risk_high = model.predict(X)[0]
    return {'risk_level': 'high' if risk_high else 'low'}
