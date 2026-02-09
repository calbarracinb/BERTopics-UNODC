from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pandas as pd

URL_RE = re.compile(r"^https?://", re.IGNORECASE)
HANDLE_RE = re.compile(r"^@[A-Za-z0-9_]+$")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_À-ÿ]+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
METRIC_RE = re.compile(r"^([0-9]+(?:[.,][0-9]+)?)([KMB])?$", re.IGNORECASE)
STATUS_ID_RE = re.compile(r"/status/(\d+)")
SIMCI_RE = re.compile(r"\bsimci\b", re.IGNORECASE)
EMOJI_URL_RE = re.compile(r"/emoji/v2/svg/([0-9a-fA-F-]+)\.svg$")

DEFAULT_NUMERIC_ENGAGEMENT_COLS = [18, 29, 31, 32]
DEFAULT_VIEWS_COL = 20
CONTENT_SCAN_START = 7
CONTENT_SCAN_END = 56
EXCLUDED_HANDLES = {"@simciupp", "@uppachuca", "@cib_uppachuca"}

NOISE_TOKENS = {
    "and",
    "·",
    "…",
    "quote",
    "show more",
    "ask grok",
    ".",
    ",",
}

PROJECT_HASHTAGS = {"#stand", "#simci"}
LOCATION_HASHTAGS_HINT = {"#buenosaires", "#argentina"}
FLAG_EMOJIS = {"🇦🇷", "🇺🇸", "🇺🇳"}


def _raw_col(idx: int) -> str:
    return f"col_{idx:02d}"


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def _looks_like_emoji_token(token: str) -> bool:
    return bool(token) and any(unicodedata.category(ch) in {"So", "Sk"} for ch in token)


def parse_metric(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return None

    # "1,1K" -> "1.1K", "1,234" -> "1234"
    if "," in text and "." not in text and text.endswith(("K", "M", "B")) and text.count(",") == 1:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")

    match = METRIC_RE.fullmatch(text)
    if not match:
        return None

    number = float(match.group(1))
    suffix = (match.group(2) or "").upper()
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return int(round(number * multiplier))


def parse_scrape_date(input_csv: Path) -> date:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", input_csv.name)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    return date.today()


def parse_tweet_date(raw_value: Any, scrape_date: date) -> pd.Timestamp:
    text = _normalize_ws(str(raw_value or ""))
    if not text:
        return pd.NaT

    for fmt in ("%b %d, %Y", "%b %d"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue

        if fmt == "%b %d":
            parsed = parsed.replace(year=scrape_date.year)
            if parsed.date() > scrape_date:
                parsed = parsed.replace(year=scrape_date.year - 1)
        return pd.Timestamp(parsed.date())

    return pd.NaT


def _col_index(col_name: str) -> int:
    match = re.search(r"(\d+)$", col_name)
    if match:
        return int(match.group(1))
    return -1


def _is_profile_url(value: str) -> bool:
    text = _normalize_ws(value)
    if not URL_RE.match(text):
        return False
    low = text.lower()
    if "/status/" in low or "/analytics" in low or "/hashtag/" in low:
        return False
    if "x.com/" in low:
        path = text.split("x.com/", 1)[1].split("?", 1)[0].strip("/")
    elif "twitter.com/" in low:
        path = text.split("twitter.com/", 1)[1].split("?", 1)[0].strip("/")
    else:
        return False
    if not path or "/" in path:
        return False
    reserved = {
        "home",
        "explore",
        "notifications",
        "messages",
        "settings",
        "search",
        "hashtag",
        "i",
    }
    return path.lower() not in reserved


def infer_column_layout(raw_df: pd.DataFrame, scrape_date: date) -> dict[str, Any]:
    stats_rows: list[dict[str, Any]] = []

    for col_name in raw_df.columns:
        series = raw_df[col_name].astype(str).str.strip()
        non_empty = series[series != ""]
        non_empty_ratio = float((series != "").mean()) if len(series) else 0.0

        if non_empty.empty:
            stats_rows.append(
                {
                    "column": col_name,
                    "column_index": _col_index(col_name),
                    "non_empty_ratio": non_empty_ratio,
                    "url_ratio": 0.0,
                    "status_ratio": 0.0,
                    "analytics_ratio": 0.0,
                    "profile_url_ratio": 0.0,
                    "handle_ratio": 0.0,
                    "date_ratio": 0.0,
                    "replying_ratio": 0.0,
                    "metric_ratio": 0.0,
                    "metric_median": 0.0,
                    "views_score": 0.0,
                }
            )
            continue

        low = non_empty.str.lower()
        metric_values = non_empty.apply(parse_metric)
        metric_ratio = float(metric_values.notna().mean())
        metric_median = float(metric_values.dropna().median()) if metric_values.notna().any() else 0.0

        stats_rows.append(
            {
                "column": col_name,
                "column_index": _col_index(col_name),
                "non_empty_ratio": non_empty_ratio,
                "url_ratio": float(non_empty.str.match(URL_RE).mean()),
                "status_ratio": float(low.str.contains("/status/", regex=False).mean()),
                "analytics_ratio": float(low.str.contains("/analytics", regex=False).mean()),
                "profile_url_ratio": float(non_empty.apply(_is_profile_url).mean()),
                "handle_ratio": float(non_empty.str.match(HANDLE_RE).mean()),
                "date_ratio": float(non_empty.apply(lambda x: pd.notna(parse_tweet_date(x, scrape_date))).mean()),
                "replying_ratio": float(low.str.startswith("replying to").mean()),
                "metric_ratio": metric_ratio,
                "metric_median": metric_median,
                "views_score": metric_ratio * (1.0 + math.log1p(max(metric_median, 0.0))),
            }
        )

    stats = pd.DataFrame(stats_rows)

    def pick(
        score_col: str,
        min_score: float = 0.0,
        exclude: set[str] | None = None,
    ) -> str | None:
        excluded = exclude or set()
        candidates = stats[~stats["column"].isin(excluded)].sort_values(
            [score_col, "non_empty_ratio"], ascending=False
        )
        if candidates.empty:
            return None
        best = candidates.iloc[0]
        if float(best[score_col]) < min_score:
            return None
        return str(best["column"])

    tweet_url_col = pick("status_ratio", min_score=0.15)
    analytics_url_col = pick("analytics_ratio", min_score=0.15, exclude={tweet_url_col} if tweet_url_col else set())
    author_handle_col = pick("handle_ratio", min_score=0.15)
    tweet_date_col = pick("date_ratio", min_score=0.15)
    replying_to_col = pick("replying_ratio", min_score=0.02)
    author_profile_col = pick(
        "profile_url_ratio",
        min_score=0.15,
        exclude={c for c in [tweet_url_col, analytics_url_col] if c},
    )
    views_col = pick("views_score", min_score=0.05)

    excluded_for_name = {c for c in [tweet_url_col, analytics_url_col, author_handle_col, tweet_date_col, replying_to_col] if c}
    author_name_col: str | None = None
    if author_handle_col:
        handle_idx = _col_index(author_handle_col)
        for delta in (-1, -2, 1, 2):
            candidate_idx = handle_idx + delta
            if candidate_idx < 0:
                continue
            candidate_col = _raw_col(candidate_idx)
            if candidate_col not in stats["column"].values:
                continue
            if candidate_col in excluded_for_name:
                continue
            row = stats.loc[stats["column"] == candidate_col].iloc[0]
            if (
                float(row["non_empty_ratio"]) >= 0.15
                and float(row["url_ratio"]) < 0.4
                and float(row["handle_ratio"]) < 0.3
                and float(row["metric_ratio"]) < 0.3
                and float(row["date_ratio"]) < 0.3
            ):
                author_name_col = candidate_col
                break

    metric_cols = (
        stats[(stats["metric_ratio"] > 0.01) & (~stats["column"].eq(views_col))]
        .sort_values("column_index")["column"]
        .tolist()
    )

    excluded_cols = {
        c
        for c in [
            tweet_url_col,
            analytics_url_col,
            author_profile_col,
            author_name_col,
            author_handle_col,
            tweet_date_col,
            replying_to_col,
            views_col,
        ]
        if c
    }
    excluded_cols.update(metric_cols)

    content_cols = [
        col
        for col in raw_df.columns
        if col not in excluded_cols and _col_index(col) >= 6
    ]

    return {
        "tweet_url_col": tweet_url_col or _raw_col(4),
        "analytics_url_col": analytics_url_col or _raw_col(19),
        "author_profile_col": author_profile_col or _raw_col(0),
        "author_name_col": author_name_col or _raw_col(2),
        "author_handle_col": author_handle_col or _raw_col(3),
        "tweet_date_col": tweet_date_col or _raw_col(5),
        "replying_to_col": replying_to_col or _raw_col(6),
        "views_col": views_col or _raw_col(DEFAULT_VIEWS_COL),
        "engagement_cols": metric_cols or [_raw_col(i) for i in DEFAULT_NUMERIC_ENGAGEMENT_COLS],
        "content_cols": content_cols or [_raw_col(i) for i in range(CONTENT_SCAN_START, CONTENT_SCAN_END + 1)],
    }


def _is_noise_fragment(fragment: str) -> bool:
    low = fragment.lower().strip()
    if not low:
        return True
    if low in NOISE_TOKENS:
        return True
    if re.fullmatch(r"and \d+ others", low):
        return True
    if URL_RE.match(fragment):
        return True
    if METRIC_RE.fullmatch(fragment.upper()):
        return True
    if low.startswith("replying to"):
        return True
    return False


def _clean_fragment(fragment: str) -> str:
    value = _normalize_ws(fragment)
    if not value:
        return ""
    if re.search(r"\band \d+ others\b", value, flags=re.IGNORECASE):
        return ""
    value = re.sub(r"\s+", " ", value).strip()
    if _is_noise_fragment(value):
        return ""
    return value


def _url_to_token(url: str) -> tuple[str, str] | None:
    value = _normalize_ws(url)
    if not URL_RE.match(value):
        return None

    low = value.lower()

    emoji_match = EMOJI_URL_RE.search(low)
    if emoji_match:
        code = emoji_match.group(1)
        try:
            return "".join(chr(int(part, 16)) for part in code.split("-")), "emoji"
        except ValueError:
            return None

    if "x.com/hashtag/" in low:
        raw_tag = value.split("/hashtag/", 1)[1].split("?", 1)[0]
        tag = unquote(raw_tag).strip()
        if tag:
            return f"#{tag}", "hashtag"
        return None

    if "x.com/" in low and "/status/" not in low:
        raw_path = value.split("x.com/", 1)[1].split("?", 1)[0].strip("/")
        if "/" in raw_path:
            return None
        reserved = {
            "",
            "hashtag",
            "i",
            "search",
            "home",
            "explore",
            "messages",
            "compose",
            "settings",
        }
        if raw_path.lower() in reserved:
            return None
        return f"@{raw_path}", "mention"

    return None


def _piece(text: str, kind: str, idx: int) -> dict[str, Any]:
    return {"text": text, "kind": kind, "idx": idx}


def _infer_kind(text: str) -> str:
    if text.startswith("#"):
        return "hashtag"
    if text.startswith("@"):
        return "mention"
    if _looks_like_emoji_token(text):
        return "emoji"
    return "text"


def _dedupe_pieces(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    seen_long_text: set[str] = set()
    emoji_counts: dict[str, int] = {}

    for piece in pieces:
        if not piece:
            continue
        text = str(piece["text"])
        kind = str(piece["kind"])
        low = text.lower()
        is_symbol_token = kind in {"hashtag", "mention", "emoji"}

        if is_symbol_token:
            if kind == "emoji":
                # Keep repeated arrows, but cap to avoid extreme noise.
                if text == "👇":
                    emoji_counts[text] = emoji_counts.get(text, 0) + 1
                    if emoji_counts[text] > 3:
                        continue
                elif low in seen_tags:
                    continue
            else:
                if low in seen_tags:
                    continue
            seen_tags.add(low)
            deduped.append(piece)
            continue

        if deduped and str(deduped[-1]["text"]).lower() == low:
            continue
        if len(low) >= 8 and low in seen_long_text:
            continue
        deduped.append(piece)
        if len(low) >= 8:
            seen_long_text.add(low)

    return deduped


def _move_first(
    pieces: list[dict[str, Any]],
    predicate,
    target_index: int,
) -> tuple[list[dict[str, Any]], bool]:
    found = next((i for i, p in enumerate(pieces) if predicate(p)), None)
    if found is None:
        return pieces, False
    token = pieces.pop(found)
    if found < target_index:
        target_index -= 1
    target_index = max(0, min(target_index, len(pieces)))
    pieces.insert(target_index, token)
    return pieces, True


def _find_first_text_index(pieces: list[dict[str, Any]], pattern: str) -> int | None:
    pattern_low = pattern.lower()
    for i, p in enumerate(pieces):
        if p["kind"] == "text" and pattern_low in str(p["text"]).lower():
            return i
    return None


def _postprocess_piece_order(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pieces:
        return pieces

    ordered = list(pieces)

    # If tweet starts with "| Con...", place #Argentina and flag before the bar segment.
    first_text_idx = next((i for i, p in enumerate(ordered) if p["kind"] == "text"), None)
    if first_text_idx is not None and str(ordered[first_text_idx]["text"]).lstrip().startswith("|"):
        ordered, _ = _move_first(ordered, lambda p: str(p["text"]).lower() == "#argentina", first_text_idx)
        first_text_idx = next((i for i, p in enumerate(ordered) if p["kind"] == "text"), None)
        if first_text_idx is not None:
            ordered, _ = _move_first(ordered, lambda p: str(p["text"]) == "🇦🇷", first_text_idx)

    # Move project hashtags (and related flags) near "proyecto".
    project_idx = _find_first_text_index(ordered, "proyecto")
    if project_idx is not None:
        insert_pos = project_idx + 1
        for token in ["#stand", "🇺🇸", "🇺🇳", "#simci"]:
            ordered, moved = _move_first(ordered, lambda p, t=token: str(p["text"]).lower() == t.lower(), insert_pos)
            if not moved:
                continue
            insert_pos += 1

    # Move location hashtag near "realizamos en".
    realiza_idx = _find_first_text_index(ordered, "realizamos en")
    if realiza_idx is not None:
        ordered, _ = _move_first(
            ordered,
            lambda p: p["kind"] == "hashtag" and str(p["text"]).lower() == "#buenosaires",
            realiza_idx + 1,
        )

    # Place mentions right after "funcionario/as de".
    func_idx = None
    for i, p in enumerate(ordered):
        if p["kind"] == "text" and re.search(r"funcionario/?as de", str(p["text"]), flags=re.IGNORECASE):
            func_idx = i
            break
    if func_idx is not None:
        mentions = [p for p in ordered if p["kind"] == "mention"]
        if mentions:
            ordered = [p for p in ordered if p["kind"] != "mention"]
            insert: list[dict[str, Any]] = []
            for j, m in enumerate(mentions[:4]):
                if j > 0:
                    insert.append(_piece("y", "text", 999))
                insert.append(m)
            ordered[func_idx + 1 : func_idx + 1] = insert

    return ordered


def _build_model_text(display_text: str) -> str:
    if not display_text:
        return ""

    text = display_text
    text = re.sub(r"^\s*[:;|,-]+\s*", "", text)
    text = re.sub(r"#[A-Za-z0-9_À-ÿ]+", lambda m: m.group(0)[1:], text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    # When mentions disappear, we can get broken connectors like "de y sus".
    text = re.sub(r"\b(de|con|para|a)\s+y\s+(?=\w)", r"\1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bde\s+para\b", "para", text, flags=re.IGNORECASE)
    text = re.sub(r"[|]+", " ", text)
    text = "".join(ch if not _looks_like_emoji_token(ch) else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_tweet_components(row: pd.Series, content_columns: list[str] | None = None) -> dict[str, Any]:
    pieces: list[dict[str, Any]] = []

    if content_columns is None:
        scan_columns = [_raw_col(idx) for idx in range(CONTENT_SCAN_START, CONTENT_SCAN_END + 1)]
    else:
        scan_columns = content_columns

    for col_name in scan_columns:
        value = _normalize_ws(str(row.get(col_name, "")))
        if not value:
            continue

        if URL_RE.match(value):
            token = _url_to_token(value)
            if token:
                token_text, token_kind = token
                pieces.append(_piece(token_text, token_kind, _col_index(col_name)))
            continue

        if parse_metric(value) is not None:
            continue

        cleaned = _clean_fragment(value)
        if cleaned:
            pieces.append(_piece(cleaned, _infer_kind(cleaned), _col_index(col_name)))

    pieces = _dedupe_pieces(pieces)
    pieces = _postprocess_piece_order(pieces)
    display_text = _normalize_ws(" ".join(str(p["text"]) for p in pieces))
    display_text = re.sub(r"\s+([,.;:!?])", r"\1", display_text)
    display_text = re.sub(r"\s+\|", " |", display_text)
    display_text = re.sub(
        r"(#STAND(?:\s+🇺🇸)?(?:\s+🇺🇳)?)\s+(#SIMCI)\s+y\s+realizamos",
        r"\1 y \2 realizamos",
        display_text,
        flags=re.IGNORECASE,
    )
    display_text = re.sub(r"\bfuncionario/as de\s+y\b", "funcionario/as de", display_text, flags=re.IGNORECASE)
    display_text = re.sub(r"\bONU Argentina\b(?=\s+para el intercambio)", "", display_text, flags=re.IGNORECASE)
    display_text = re.sub(r"\s+y\s+para\s+", " para ", display_text, flags=re.IGNORECASE)
    display_text = re.sub(r"\s{2,}", " ", display_text)
    display_text = display_text.strip()

    model_text = _build_model_text(display_text)
    mentions = sorted(set(MENTION_RE.findall(display_text)))
    hashtags = sorted(set(HASHTAG_RE.findall(display_text)))
    emojis = sorted(set(str(p["text"]) for p in pieces if p["kind"] == "emoji"))

    return {
        "tweet_text_display": display_text,
        "tweet_text_model": model_text,
        "mentions": mentions,
        "hashtags": hashtags,
        "emojis": emojis,
        "contains_simci": bool(SIMCI_RE.search(display_text)),
    }


def extract_status_id(tweet_url: str) -> str | None:
    match = STATUS_ID_RE.search(tweet_url or "")
    if match:
        return match.group(1)
    return None


def extract_engagement(
    row: pd.Series,
    engagement_cols: list[str] | None = None,
    views_col: str | None = None,
) -> dict[str, Any]:
    metric_cols = engagement_cols or [_raw_col(idx) for idx in DEFAULT_NUMERIC_ENGAGEMENT_COLS]
    counts: list[tuple[str, int]] = []
    for col_name in metric_cols:
        value = parse_metric(row.get(col_name, ""))
        if value is not None:
            counts.append((col_name, value))

    views = parse_metric(row.get(views_col or _raw_col(DEFAULT_VIEWS_COL), ""))

    likes = counts[-1][1] if counts else 0
    reposts = counts[-2][1] if len(counts) >= 2 else 0
    comments = counts[-3][1] if len(counts) >= 3 else 0

    if len(counts) >= 3:
        confidence = "high"
    elif len(counts) == 2:
        confidence = "medium"
    elif len(counts) == 1:
        confidence = "low"
    else:
        confidence = "none"

    raw_positions = ",".join(str(col_name) for col_name, _ in counts)
    raw_values = ",".join(str(value) for _, value in counts)

    return {
        "likes": likes,
        "reposts": reposts,
        "comments": comments,
        "views": views or 0,
        "engagement_signal_count": len(counts),
        "engagement_confidence": confidence,
        "engagement_raw_positions": raw_positions,
        "engagement_raw_values": raw_values,
    }


def build_column_profile(df: pd.DataFrame, original_headers: list[str]) -> pd.DataFrame:
    profile_rows: list[dict[str, Any]] = []
    total_rows = len(df)

    for idx, original_name in enumerate(original_headers):
        col_name = _raw_col(idx)
        series = df[col_name].astype(str).str.strip()
        non_empty = series[series != ""]
        non_empty_count = len(non_empty)

        if non_empty_count == 0:
            inferred_type = "empty"
            top_examples = ""
        else:
            url_ratio = (non_empty.str.match(URL_RE).sum()) / non_empty_count
            num_ratio = non_empty.apply(lambda x: parse_metric(x) is not None).sum() / non_empty_count
            handle_ratio = non_empty.str.match(HANDLE_RE).sum() / non_empty_count

            if url_ratio > 0.9:
                inferred_type = "url"
            elif num_ratio > 0.9:
                inferred_type = "metric"
            elif handle_ratio > 0.8:
                inferred_type = "handle"
            else:
                inferred_type = "text/mixed"

            top_examples = " | ".join(non_empty.value_counts().head(5).index.tolist())

        profile_rows.append(
            {
                "column_index": idx,
                "raw_column_name": original_name,
                "new_column_name": col_name,
                "non_empty_count": non_empty_count,
                "non_empty_ratio": round(non_empty_count / total_rows if total_rows else 0.0, 4),
                "inferred_type": inferred_type,
                "top_examples": top_examples,
            }
        )

    return pd.DataFrame(profile_rows)


def _is_excluded_handle(author_handle: str, replying_to_raw: str, tweet_text_display: str, mentions: list[str]) -> bool:
    haystack = " ".join([author_handle, replying_to_raw, tweet_text_display, " ".join(mentions)]).lower()
    return any(handle in haystack for handle in EXCLUDED_HANDLES)


def _normalize_for_rules(text: str) -> str:
    low = _strip_accents(text.lower())
    return re.sub(r"\s+", " ", low).strip()


def infer_simci_stance(text: str) -> str:
    normalized = _normalize_for_rules(text)
    if "simci" not in normalized:
        return "no_menciona"

    score = 0

    if re.search(r"(metodolog|simci).{0,40}(riguros|transparen|confiab|precis|cientif)", normalized):
        score += 2
    if re.search(r"(simci|metodolog).{0,40}(mal|fals|equivoc|errad|sesgad|sobreestim|inflad|manipul)", normalized):
        score -= 2

    if re.search(r"(criticas?|cuestionamientos?).{0,45}(simci|metodolog).{0,45}no (son|es|estan|esta)", normalized):
        score += 2
    if re.search(r"(simci|metodolog).{0,50}no (esta|estan|es|son) (mal|equivocad|fals)", normalized):
        score += 2
    if re.search(
        r"(argument(ar|a|an)|decir|afirman?|sostienen?).{0,120}(simci|metodolog).{0,80}(esta|estan|es|son) mal",
        normalized,
    ) and re.search(r"(no les va a salir bien|no son ciertas?|mentiras?|falso|falsa)", normalized):
        score += 4
    if re.search(r"(simci).{0,30}(miente|mentira|falso|falsa|manipula|sobreestima)", normalized):
        score -= 2

    positive_tokens = ["defiende", "respalda", "apoya", "riguroso", "transparente", "confiable", "preciso"]
    negative_tokens = ["mal", "falso", "falsa", "errado", "equivocado", "sesgado", "inflado", "sobreestimado"]

    window_matches = re.finditer(r".{0,120}simci.{0,120}", normalized)
    for match in window_matches:
        window = match.group(0)
        score += sum(1 for token in positive_tokens if token in window)
        score -= sum(1 for token in negative_tokens if token in window)

    if score >= 1:
        return "pro_simci"
    if score <= -1:
        return "anti_simci"
    return "neutral"


def _adjust_sentiment(base_sentiment: str, simci_stance: str) -> str:
    if simci_stance == "pro_simci":
        return "POS"
    if simci_stance == "anti_simci":
        return "NEG"
    if base_sentiment in {"POS", "NEU", "NEG"}:
        return base_sentiment
    return "NEU"


def _dedupe_clean_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    if clean_df.empty:
        return clean_df

    out = clean_df.copy()
    out["tweet_id"] = out["tweet_id"].fillna("").astype(str).str.strip()
    out["tweet_url"] = out["tweet_url"].fillna("").astype(str).str.strip()
    out["dedupe_key"] = out["tweet_id"]
    out.loc[out["dedupe_key"] == "", "dedupe_key"] = out["tweet_url"]
    out = out[out["dedupe_key"] != ""].copy()

    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col_name, asc in [("tweet_date", False), ("views", False), ("likes", False), ("tweet_id", False)]:
        if col_name in out.columns:
            sort_cols.append(col_name)
            ascending.append(asc)
    if sort_cols:
        out.sort_values(by=sort_cols, ascending=ascending, inplace=True, na_position="last")

    out.drop_duplicates(subset=["dedupe_key"], keep="first", inplace=True)
    out.drop(columns=["dedupe_key"], inplace=True, errors="ignore")
    return out


def _clean_single_dataset(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_csv(input_path, dtype=str).fillna("")
    original_headers = list(raw_df.columns)
    raw_df.columns = [_raw_col(i) for i in range(raw_df.shape[1])]

    scrape_date = parse_scrape_date(input_path)
    layout = infer_column_layout(raw_df, scrape_date=scrape_date)

    profile_df = build_column_profile(raw_df, original_headers)
    profile_df["source_file"] = input_path.name

    role_map = {
        layout.get("tweet_url_col"): "tweet_url",
        layout.get("analytics_url_col"): "analytics_url",
        layout.get("author_profile_col"): "author_profile_url",
        layout.get("author_name_col"): "author_name",
        layout.get("author_handle_col"): "author_handle",
        layout.get("tweet_date_col"): "tweet_date_raw",
        layout.get("replying_to_col"): "replying_to_raw",
        layout.get("views_col"): "views",
    }
    role_map = {k: v for k, v in role_map.items() if k}
    profile_df["layout_role"] = profile_df["new_column_name"].map(role_map).fillna("")
    profile_df["layout_group"] = ""
    profile_df.loc[profile_df["new_column_name"].isin(layout.get("engagement_cols", [])), "layout_group"] = "engagement"
    profile_df.loc[profile_df["new_column_name"].isin(layout.get("content_cols", [])), "layout_group"] = "content"

    cleaned_rows: list[dict[str, Any]] = []
    for _, row in raw_df.iterrows():
        tweet_url = _normalize_ws(str(row.get(layout["tweet_url_col"], "")))
        author_handle = _normalize_ws(str(row.get(layout["author_handle_col"], "")))
        replying_to_raw = _normalize_ws(str(row.get(layout["replying_to_col"], "")))
        tweet_date_raw = _normalize_ws(str(row.get(layout["tweet_date_col"], "")))

        components = extract_tweet_components(row, content_columns=layout.get("content_cols"))
        if _is_excluded_handle(
            author_handle=author_handle,
            replying_to_raw=replying_to_raw,
            tweet_text_display=components["tweet_text_display"],
            mentions=components["mentions"],
        ):
            continue

        metrics = extract_engagement(
            row,
            engagement_cols=layout.get("engagement_cols"),
            views_col=layout.get("views_col"),
        )
        cleaned_rows.append(
            {
                "source_file": input_path.name,
                "tweet_id": extract_status_id(tweet_url),
                "tweet_url": tweet_url,
                "analytics_url": _normalize_ws(str(row.get(layout["analytics_url_col"], ""))),
                "author_profile_url": _normalize_ws(str(row.get(layout["author_profile_col"], ""))),
                "author_name": _normalize_ws(str(row.get(layout["author_name_col"], ""))),
                "author_handle": author_handle,
                "tweet_date_raw": tweet_date_raw,
                "tweet_date": parse_tweet_date(tweet_date_raw, scrape_date),
                "replying_to_raw": replying_to_raw,
                "tweet_text": components["tweet_text_display"],  # backward-compat field
                "tweet_text_display": components["tweet_text_display"],
                "tweet_text_model": components["tweet_text_model"],
                "mentions": "|".join(components["mentions"]),
                "hashtags": "|".join(components["hashtags"]),
                "emojis": "".join(components["emojis"]),
                "contains_simci": components["contains_simci"],
                "likes": metrics["likes"],
                "reposts": metrics["reposts"],
                "comments": metrics["comments"],
                "views": metrics["views"],
                "engagement_signal_count": metrics["engagement_signal_count"],
                "engagement_confidence": metrics["engagement_confidence"],
                "engagement_raw_positions": metrics["engagement_raw_positions"],
                "engagement_raw_values": metrics["engagement_raw_values"],
            }
        )

    clean_df = pd.DataFrame(cleaned_rows)
    if clean_df.empty:
        return clean_df, profile_df
    clean_df = clean_df[clean_df["tweet_url"] != ""].copy()
    clean_df = _dedupe_clean_df(clean_df)
    return clean_df, profile_df


def clean_datasets(
    input_csvs: list[str | Path] | tuple[str | Path, ...],
    output_clean_csv: str | Path,
    output_profile_csv: str | Path,
) -> pd.DataFrame:
    input_paths = [Path(p) for p in input_csvs]
    if not input_paths:
        raise ValueError("At least one input CSV is required.")
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    clean_parts: list[pd.DataFrame] = []
    profile_parts: list[pd.DataFrame] = []

    for input_path in input_paths:
        clean_df, profile_df = _clean_single_dataset(input_path)
        clean_parts.append(clean_df)
        profile_parts.append(profile_df)

    combined_clean = pd.concat(clean_parts, ignore_index=True) if clean_parts else pd.DataFrame()
    combined_clean = _dedupe_clean_df(combined_clean)
    if not combined_clean.empty:
        combined_clean.sort_values(by=["tweet_date", "tweet_id"], ascending=[False, False], inplace=True, na_position="last")

    combined_profile = pd.concat(profile_parts, ignore_index=True) if profile_parts else pd.DataFrame()
    if not combined_profile.empty and "source_file" in combined_profile.columns:
        combined_profile.sort_values(by=["source_file", "column_index"], inplace=True)

    output_clean = Path(output_clean_csv)
    output_profile = Path(output_profile_csv)
    output_clean.parent.mkdir(parents=True, exist_ok=True)
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    combined_clean.to_csv(output_clean, index=False)
    combined_profile.to_csv(output_profile, index=False)
    return combined_clean


def clean_dataset(input_csv: str | Path, output_clean_csv: str | Path, output_profile_csv: str | Path) -> pd.DataFrame:
    return clean_datasets([input_csv], output_clean_csv=output_clean_csv, output_profile_csv=output_profile_csv)


def run_sentiment(
    input_clean_csv: str | Path,
    output_sentiment_csv: str | Path,
    text_column: str = "tweet_text_model",
) -> pd.DataFrame:
    try:
        from pysentimiento import create_analyzer
    except ImportError as exc:
        raise RuntimeError("Missing dependency `pysentimiento`. Run `pip install -r requirements.txt`.") from exc

    input_path = Path(input_clean_csv)
    output_path = Path(output_sentiment_csv)

    df = pd.read_csv(input_path, dtype={"tweet_id": "string"}).fillna("")
    sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
    emotion_analyzer = create_analyzer(task="emotion", lang="es")

    sentiment_labels: list[str] = []
    sentiment_pos: list[float] = []
    sentiment_neu: list[float] = []
    sentiment_neg: list[float] = []

    emotion_labels: list[str] = []
    emotion_prob_dicts: list[dict[str, float]] = []

    simci_stances: list[str] = []
    sentiment_adjusted: list[str] = []

    for _, row in df.iterrows():
        display_text = _normalize_ws(str(row.get("tweet_text_display", "")))
        model_text = _normalize_ws(str(row.get(text_column, ""))) or display_text

        if not model_text:
            sentiment = "NEU"
            sentiment_proba = {"POS": 0.0, "NEU": 1.0, "NEG": 0.0}
            emotion = "others"
            emotion_proba: dict[str, float] = {"others": 1.0}
        else:
            try:
                s_pred = sentiment_analyzer.predict(model_text)
                sentiment = str(s_pred.output)
                sentiment_proba = {k: float(v) for k, v in (s_pred.probas or {}).items()}
            except Exception:
                sentiment = "NEU"
                sentiment_proba = {"POS": 0.0, "NEU": 1.0, "NEG": 0.0}

            try:
                e_pred = emotion_analyzer.predict(model_text)
                emotion = str(e_pred.output)
                emotion_proba = {k: float(v) for k, v in (e_pred.probas or {}).items()}
            except Exception:
                emotion = "others"
                emotion_proba = {"others": 1.0}

        stance = infer_simci_stance(display_text)
        adjusted = _adjust_sentiment(sentiment, stance)

        sentiment_labels.append(sentiment)
        sentiment_pos.append(float(sentiment_proba.get("POS", 0.0)))
        sentiment_neu.append(float(sentiment_proba.get("NEU", 0.0)))
        sentiment_neg.append(float(sentiment_proba.get("NEG", 0.0)))

        emotion_labels.append(emotion)
        emotion_prob_dicts.append(emotion_proba)

        simci_stances.append(stance)
        sentiment_adjusted.append(adjusted)

    df["sentiment"] = sentiment_labels
    df["sentiment_positive"] = sentiment_pos
    df["sentiment_neutral"] = sentiment_neu
    df["sentiment_negative"] = sentiment_neg
    df["sentiment_label_es"] = df["sentiment"].map({"POS": "Positivo", "NEU": "Neutral", "NEG": "Negativo"})

    df["emotion"] = emotion_labels
    emotion_label_map = {
        "joy": "Alegría",
        "sadness": "Tristeza",
        "anger": "Enojo",
        "fear": "Miedo",
        "surprise": "Sorpresa",
        "disgust": "Asco",
        "others": "Otros",
    }
    df["emotion_label_es"] = df["emotion"].map(emotion_label_map).fillna(df["emotion"])

    all_emotion_keys = sorted({k for probs in emotion_prob_dicts for k in probs.keys()})
    for key in all_emotion_keys:
        safe_key = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        column_name = f"emotion_{safe_key}" if safe_key else "emotion_other"
        df[column_name] = [float(probs.get(key, 0.0)) for probs in emotion_prob_dicts]

    df["simci_stance"] = simci_stances
    df["simci_stance_label_es"] = df["simci_stance"].map(
        {
            "pro_simci": "A Favor De SIMCI",
            "anti_simci": "En Contra De SIMCI",
            "neutral": "Neutral Sobre SIMCI",
            "no_menciona": "No Menciona SIMCI",
        }
    )
    df["sentiment_simci_adjusted"] = sentiment_adjusted
    df["sentiment_simci_adjusted_label_es"] = df["sentiment_simci_adjusted"].map(
        {"POS": "Positivo", "NEU": "Neutral", "NEG": "Negativo"}
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def run_full_pipeline(
    raw_csv: str | Path | list[str | Path] | tuple[str | Path, ...],
    clean_csv: str | Path = "data/processed/tweets_clean.csv",
    profile_csv: str | Path = "data/processed/column_profile.csv",
    sentiment_csv: str | Path = "data/processed/tweets_sentiment.csv",
) -> dict[str, Any]:
    if isinstance(raw_csv, (list, tuple)):
        inputs = list(raw_csv)
    else:
        inputs = [raw_csv]
    clean_df = clean_datasets(inputs, clean_csv, profile_csv)
    sentiment_df = run_sentiment(clean_csv, sentiment_csv, text_column="tweet_text_model")
    return {
        "raw_input": ",".join(str(p) for p in inputs),
        "clean_output": str(clean_csv),
        "profile_output": str(profile_csv),
        "sentiment_output": str(sentiment_csv),
        "rows_clean": int(len(clean_df)),
        "rows_sentiment": int(len(sentiment_df)),
    }
