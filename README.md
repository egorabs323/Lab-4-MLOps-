# Лабораторная работа 4: MLOps — вариант 14

**Задача:** состав бетонной смеси и возраст (8 признаков) → прочность бетона в МПа (регрессия).
Источник: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength — Yeh, 1998; UCI, CC BY 4.0.

Этот проект адаптирован из методического примера с Wine: `Ridge` и `RandomForestRegressor`, выбор по наименьшему **validation MAE**, дополнительная метрика RMSE. Исходный датасет и фактические результаты не включены: их необходимо получить самостоятельно при запуске. Все команды выполняйте **из корня проекта** в Ubuntu.

## 1. Подготовка

```bash
cd ~/concrete_mlops_lab
python3 --version
sudo docker version
sudo docker compose version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > requirements.lock.txt
```

В случае отсутствия Docker установите его по **разделу 4.2 методички**, затем проверьте `sudo docker run --rm hello-world`.

## 2. Данные и разбиение

```bash
python src/prepare_data.py
sha256sum data/raw/Concrete_Data.xls
cat data/split.json
```

Скрипт получает оригинальный ZIP UCI, извлекает `Concrete_Data.xls` в `data/raw/`, сохраняет SHA-256 и **однократно** создаёт разбиение 60/20/20 (`train` = 618, `validation` = 206, `test` = 206; `random_state=42`). Для регрессии `stratify` не используется. Повторные запуски используют те же индексы. Если сайт UCI недоступен из Python, скачайте ZIP на странице UCI, извлеките **оригинальный** `Concrete_Data.xls` вручную в `data/raw/` и повторите команду. Не переименовывайте колонки в исходном XLS; это делается в коде без изменения оригинала.

## 3. Запуск MLflow

Откройте **терминал A**, не закрывайте его до окончания работы:

```bash
cd ~/concrete_mlops_lab
source .venv/bin/activate
mkdir -p mlflow-data/artifacts
mlflow server \
  --backend-store-uri sqlite:///mlflow-data/mlflow.db \
  --artifacts-destination ./mlflow-data/artifacts \
  --serve-artifacts \
  --host 127.0.0.1 \
  --port 5000
```

Откройте **терминал B**:

```bash
cd ~/concrete_mlops_lab
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
curl -f http://127.0.0.1:5000/health
```

В браузере откройте http://127.0.0.1:5000. Выберите эксперимент `lab4-concrete-strength` после первого запуска.

## 4. Восемь экспериментов

Каждую команду запускайте **после завершения предыдущей**:

```bash
python src/train.py --algorithm ridge --alpha 0.1
python src/train.py --algorithm ridge --alpha 1
python src/train.py --algorithm ridge --alpha 10
python src/train.py --algorithm rf --trees 50 --depth 5
python src/train.py --algorithm rf --trees 100 --depth 5
python src/train.py --algorithm rf --trees 200 --depth 5
python src/train.py --algorithm rf --trees 100 --depth 3
python src/train.py --algorithm rf --trees 100 --depth 12
```

Это план: 3 варианта Ridge; при фиксированной глубине 5 сравните 50/100/200 деревьев; при фиксированных 100 деревьях сравните глубину 3/5/12. Такой дизайн позволяет отдельно оценить влияние каждого параметра, чего нельзя сделать, меняя их одновременно.

Откройте MLflow → **Experiments → lab4-concrete-strength**, отсортируйте по `val_mae` **по возрастанию**. Проверьте `val_rmse`, `train_mae`, `fit_seconds`, теги, график `validation_actual_vs_predicted.png` и `run_id`. Чем меньше `val_mae`, тем лучше. **Не просматривайте test до выбора champion.**

## 5. Регистрация моделей

Зарегистрируйте **одну Ridge с лучшим alpha** и **один RandomForest с лучшими trees/depth**, найденными выше. Замените указанные значения реальными:

```bash
python src/train.py --algorithm ridge --alpha 1 --register
python src/train.py --algorithm rf --trees 100 --depth 12 --register
```

В терминале будут строки `REGISTERED_VERSION=`. В MLflow → **Models → concrete-strength-regressor** проверьте две зарегистрированные версии. Например, это могут быть 1 и 2, но при повторах номера могут быть другими.

Назначьте alias **champion** той зарегистрированной версии, у которой **меньше validation MAE**. Ниже `1` только пример:

```bash
python src/promote.py --version 1
```

**Запишите два фактических номера:** `BEST_VERSION` (исходный champion, который пойдёт в `v1`) и `OTHER_VERSION` (другая зарегистрированная модель для демонстрации обновления).

## 6. Экспорт и один итоговый test

```bash
python src/export_model.py
python src/evaluate.py
cat exported_model/metadata.json
cat test_metrics.json
cat sample_request.json
```

`export_model.py` загружает текущий alias `champion`, закрепляет конкретный номер Registry-версии в `metadata.json` и записывает модель с Pipeline в `exported_model/`. Формируются `sample_request.json` (3 test-строки) и `expected_predictions.json` (ответы экспортированной модели). `evaluate.py` один раз вычисляет итоговые `test_mae`, `test_rmse` на отложенных 206 строках. Не подбирайте гиперпараметры по результатам test.

## 7. Docker-образ v1 и ручной запуск

```bash
sudo docker build -t concrete-api:v1 .
sudo docker image inspect concrete-api:v1 --format '{{.Id}}'
sudo docker run -d --name concrete-api -p 127.0.0.1:8000:8000 concrete-api:v1
sudo docker ps
sudo docker logs concrete-api
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/metadata
curl -f -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @sample_request.json
python src/check_api.py
```

В браузере откройте http://127.0.0.1:8000/docs → POST `/predict` → Try it out; вставьте JSON из `sample_request.json`. Скрипт `check_api.py` проверяет совпадение числовых предсказаний с экспортированной моделью, HTTP 200 на корректных запросах и HTTP 422 на ошибочных (нет признаков, пустой массив, лишнее поле, неверный тип).

## 8. Обновление до v2

В команде `promote.py` замените `OTHER_VERSION` на второй фактический номер:

```bash
python src/promote.py --version OTHER_VERSION
python src/export_model.py
sudo docker build -t concrete-api:v2 .
sudo docker stop concrete-api
sudo docker rm concrete-api
sudo docker run -d --name concrete-api -p 127.0.0.1:8000:8000 concrete-api:v2
curl -f http://127.0.0.1:8000/metadata
python src/check_api.py
```

Убедитесь, что `model_version` изменился. Смысл шага — показать обновление **без повторного обучения**, а не то, что модель `v2` лучше.

## 9. Откат к v1

Замените `BEST_VERSION` на реальный номер исходного champion:

```bash
sudo docker stop concrete-api
sudo docker rm concrete-api
sudo docker run -d --name concrete-api -p 127.0.0.1:8000:8000 concrete-api:v1
python src/promote.py --version BEST_VERSION
python src/export_model.py
curl -f http://127.0.0.1:8000/metadata
python src/check_api.py
```

Откат выполняется запуском **сохранённого v1-образа**: без нового обучения и без пересборки v1. Повторный `export_model.py` нужен **только для восстановления локальных проверочных JSON** после экспорта v2 — он не меняет сохранённый образ.

## 10. Docker Compose и healthcheck

Убедитесь, что champion возвращён на исходную версию и выполнен повторный экспорт из шага 9. Освободите порт 8000:

```bash
sudo docker stop concrete-api
sudo docker rm concrete-api
sudo docker compose up -d --build
sudo docker compose ps
sudo docker compose logs --tail 50
curl -f http://127.0.0.1:8000/health
python src/check_api.py
sudo docker compose stats --no-stream
sudo docker compose down
```

В `compose.yaml` указан отдельный тег `concrete-api:compose-demo`, поэтому сборка **не перезаписывает** сохранённые `concrete-api:v1` и `concrete-api:v2`. У сервиса должен появиться статус `(healthy)` после успешного healthcheck.

## 11. Что приложить к отчёту

- Вариант 14: цель, источник и лицензия UCI; 8 признаков, целевая переменная, единицы.
- Скриншоты структуры проекта, версий Python/MLflow/sklearn, SHA-256 данных и фиксированного split.
- Не менее 6 успешных MLflow runs, таблица параметров, `val_mae` и `val_rmse`; отдельный анализ `trees` и `depth`.
- Две Registry-версии модели, alias `champion`, выбранный `run_id` и объяснение выбора.
- Экспорт модели, итоговые `test_mae`, `test_rmse`, API-контракт с кг/м³, днями и МПа.
- Скриншоты HTTP 200/422, проверки совпадения API с моделью, запуска v1 и v2, отката на v1.
- Сборка Docker (тег и image ID), успешный Compose `healthy`, пример логов и времени предсказания.
- Итоговый вывод, источники и код. Архив исходников должен включать Dockerfile, compose.yaml, requirements.txt, README и код. Для воспроизведения храните оригинальный `Concrete_Data.xls`, `data/split.json`, `requirements.lock.txt` и исходный архив или отдельную резервную копию `mlflow-data/`.

## Устройство проекта

```text
concrete_mlops_lab/
├── src/
│   ├── data_utils.py      # единый контракт набора, чтение и разбиение
│   ├── prepare_data.py    # загрузка оригинального XLS и заморозка split
│   ├── train.py           # Ridge / RandomForest, MLflow
│   ├── promote.py         # alias champion
│   ├── export_model.py    # экспорт конкретной версии + JSON-примеры
│   ├── evaluate.py        # финальный независимый test
│   ├── app.py             # FastAPI /predict /metadata /health
│   └── check_api.py       # HTTP 200, 422 и совпадение с моделью
├── data/raw/              # оригинальный Concrete_Data.xls после подготовки
├── data/split.json        # появится после prepare_data.py
├── mlflow-data/           # появится после запуска MLflow
├── exported_model/        # появится после экспорта
├── screenshots/
├── requirements.txt
├── requirements.lock.txt  # создать в Ubuntu с pip freeze
├── Dockerfile
├── compose.yaml
├── .dockerignore
└── README.md
```

## Контракт API

Все 8 числовых полей обязательны; названия используются **в нижнем регистре с подчёркиваниями**. Количество цемента, шлака, золы, воды, пластификатора и заполнителей выражается в **кг на 1 м³ смеси**, возраст — **в днях**. Выход — список `predictions` в **МПа**. Диапазон возраста из набора UCI: 1–365 дней; API отклоняет значения вне диапазона.

Пример ручного запроса:

```json
{
  "instances": [
    {
      "cement": 540,
      "blast_furnace_slag": 0,
      "fly_ash": 0,
      "water": 162,
      "superplasticizer": 2.5,
      "coarse_aggregate": 1040,
      "fine_aggregate": 676,
      "age": 28
    }
  ]
}
```

**Важно:** скрипты прошли синтаксическую проверку; полноценный запуск и метрики нужно получить в вашей Ubuntu с установленными зависимостями, локальным MLflow и Docker. В архиве нет выдуманных результатов или готовой обученной модели.
