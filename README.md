# SIMCI X Tweets Pipeline + Dashboard

Pipeline para:
- limpiar y renombrar uno o varios scrapes crudos de X (`x-YYYY-MM-DD.csv`)
- normalizar métricas (`5.6K` -> `5600`, `1.1M` -> `1100000`)
- correr análisis de sentimiento en español
- correr análisis de emociones
- ajustar un sentimiento objetivo hacia SIMCI (pro/anti/neutral)
- visualizar resultados en dashboard interactivo

## 1) Instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Ejecutar pipeline completo

```bash
python3 run_pipeline.py --input x-2026-02-08.csv
```

También puedes consolidar múltiples archivos y deduplicar por `tweet_id`/`tweet_url`:

```bash
python3 run_pipeline.py --input x-2026-02-08.csv x-2026-02-09.csv
```

O usar glob:

```bash
python3 run_pipeline.py --input-glob "x-2026-02-*.csv"
```

Archivos generados:
- `data/processed/column_profile.csv`: perfil profundo de columnas por archivo fuente
- `data/processed/tweets_clean.csv`: dataset limpio y renombrado
- `data/processed/tweets_sentiment.csv`: dataset final con sentimiento, emociones y ajuste hacia SIMCI

## 3) Abrir dashboard

```bash
streamlit run dashboard.py
```

En el dashboard:
- puedes elegir entre `Sentimiento general` y `Sentimiento hacia SIMCI (ajustado)`
- puedes ver resultados sin ponderación o ponderados por `vistas`, `reposts` o `likes`
- la evolución del sentimiento se muestra en formato apilado y con granularidad `semanal` o `mensual`
- incluye un gráfico adicional de acumulado por sentimiento
- el filtro de fechas es opcional (no se aplica por defecto)
- puedes filtrar autores desde sidebar y también seleccionando barras del gráfico de autores
- se muestran métricas totales y promedios
- el formato numérico de visualización usa convención en español (`3.500`, `12,7`)

## Notas de limpieza / inferencia

- El CSV viene con columnas CSS y estructura variable por tipo de tweet (normal, reply, quote, con/ sin media) y por fecha de scrape.
- El pipeline detecta dinámicamente las columnas clave (`tweet_url`, fecha, usuario, views y señales de engagement).
- `likes`, `reposts` y `comments` se infieren por orden de señales numéricas disponibles en cada fila.
- Se incluye `engagement_confidence` para auditar qué tan confiable es la asignación por fila.
- El texto para visualización (`tweet_text_display`) se reconstruye uniendo fragmentos + hashtags/menciones/emojis extraídos de URLs de X.
- El texto para modelo (`tweet_text_model`) aplica limpieza adicional para inferencia.
- Se excluyen tweets del usuario `@simciupp` y menciones directas a `@simciupp`.
