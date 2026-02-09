from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from src.pipeline import clean_datasets, run_sentiment


def format_number_es(value: float | int, decimals: int = 0) -> str:
    fmt = f"{{:,.{decimals}f}}"
    text = fmt.format(float(value))
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def log_step(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Limpia el scrape de X, estandariza métricas y ejecuta sentimiento/emociones en español."
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=None,
        help="Una o varias rutas de CSV crudo exportado desde el scraper.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Patrón glob opcional para cargar múltiples archivos (ej: 'x-2026-02-*.csv').",
    )
    parser.add_argument(
        "--clean-output",
        type=str,
        default="data/processed/tweets_clean.csv",
        help="Ruta de salida para el CSV limpio.",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="data/processed/column_profile.csv",
        help="Ruta de salida para el perfil de columnas.",
    )
    parser.add_argument(
        "--sentiment-output",
        type=str,
        default="data/processed/tweets_sentiment.csv",
        help="Ruta de salida para el CSV final con sentimiento y emociones.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_candidates: list[Path] = []
    if args.input:
        input_candidates.extend(Path(p) for p in args.input)
    if args.input_glob:
        input_candidates.extend(sorted(Path(".").glob(args.input_glob)))
    if not input_candidates:
        input_candidates = [Path("x-2026-02-08.csv")]

    seen: set[Path] = set()
    input_paths: list[Path] = []
    for path in input_candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        input_paths.append(path)

    if not input_paths:
        raise FileNotFoundError("No input files were provided.")
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    clean_output = Path(args.clean_output)
    profile_output = Path(args.profile_output)
    sentiment_output = Path(args.sentiment_output)

    start_total = time.perf_counter()

    log_step("Inicio del pipeline.")
    log_step(f"Entradas detectadas: {format_number_es(len(input_paths))} archivo(s).")
    for idx, input_path in enumerate(input_paths, start=1):
        log_step(f"  - [{idx}] {input_path.resolve()}")

    log_step("Paso 1/2: limpieza, consolidación de texto y perfil de columnas.")
    start_clean = time.perf_counter()
    clean_df = clean_datasets(
        input_csvs=input_paths,
        output_clean_csv=clean_output,
        output_profile_csv=profile_output,
    )
    clean_elapsed = time.perf_counter() - start_clean
    log_step(
        "Paso 1/2 completado: "
        f"{format_number_es(len(clean_df))} filas limpias en {format_number_es(clean_elapsed, 1)} s."
    )

    log_step("Paso 2/2: análisis de sentimiento, emociones y ajuste hacia SIMCI.")
    start_sent = time.perf_counter()
    sentiment_df = run_sentiment(
        input_clean_csv=clean_output,
        output_sentiment_csv=sentiment_output,
    )
    sent_elapsed = time.perf_counter() - start_sent
    log_step(
        "Paso 2/2 completado: "
        f"{format_number_es(len(sentiment_df))} filas analizadas en {format_number_es(sent_elapsed, 1)} s."
    )

    total_elapsed = time.perf_counter() - start_total
    log_step("Pipeline finalizado.")
    print("")
    print("Resumen:")
    print("- Entradas:")
    for input_path in input_paths:
        print(f"  - {input_path.resolve()}")
    print(f"- CSV limpio: {clean_output.resolve()}")
    print(f"- Perfil de columnas: {profile_output.resolve()}")
    print(f"- CSV final: {sentiment_output.resolve()}")
    print(f"- Filas finales: {format_number_es(len(sentiment_df))}")
    print(f"- Tiempo total: {format_number_es(total_elapsed, 1)} s")


if __name__ == "__main__":
    main()
