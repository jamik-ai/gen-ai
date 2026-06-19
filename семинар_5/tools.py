"""
Инструменты макро-агента: get_fx_rate, get_key_rate, get_inflation,
get_unemployment, calculate + новый compare_periods (ДЗ5, задание 1).
"""
from __future__ import annotations

import csv
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date as _date
from datetime import datetime
from pathlib import Path

import sympy

DATA_DIR = Path(__file__).resolve().parent / "data"

CBR_FX_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
CBR_KEYIND_URL = "https://www.cbr.ru/key-indicators/"

_TIMEOUT_SEC = 6
_UA = "Mozilla/5.0 (seminar-5-agent)"


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as r:
        return r.read()


def _parse_date(s: str | None) -> _date:
    if s is None:
        return _date.today()
    if isinstance(s, _date):
        return s
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Неизвестный формат даты: {s!r}")


# ===========================================================================
# 1. Курс валюты ЦБ
# ===========================================================================

def get_fx_rate(currency: str = "USD", on_date: str | None = None) -> dict:
    d = _parse_date(on_date)
    currency = currency.upper()

    try:
        q = urllib.parse.urlencode({"date_req": d.strftime("%d/%m/%Y")})
        xml_bytes = _http_get(f"{CBR_FX_URL}?{q}")
        xml_text = xml_bytes.decode("windows-1251", errors="replace")
        root = ET.fromstring(xml_text)
        for val in root.findall("Valute"):
            if val.findtext("CharCode") == currency:
                nominal = int(val.findtext("Nominal") or 1)
                raw = (val.findtext("Value") or "").replace(",", ".")
                rate = float(raw) / nominal
                return {
                    "currency": currency,
                    "date": d.isoformat(),
                    "rate": round(rate, 4),
                    "source": "cbr_live",
                }
        return _fx_fallback(currency, d, reason=f"Валюты {currency} нет в ответе ЦБ.")
    except (urllib.error.URLError, TimeoutError, ET.ParseError, ValueError) as e:
        return _fx_fallback(
            currency, d, reason=f"Сбой живого запроса: {type(e).__name__}: {e}"
        )


def _fx_fallback(currency: str, d: _date, *, reason: str) -> dict:
    path = DATA_DIR / "fx_benchmark.csv"
    best = None
    best_delta = None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["currency"] != currency:
                continue
            row_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            delta = abs((row_date - d).days)
            if best is None or delta < best_delta:
                best = row
                best_delta = delta
    if best is None:
        return {"error": f"нет запасных данных для {currency}"}
    return {
        "currency": currency,
        "date": best["date"],
        "rate": float(best["rate"]),
        "source": "fallback_csv",
        "reason": reason,
    }


# ===========================================================================
# 2. Ключевая ставка ЦБ
# ===========================================================================

_KEY_RATE_RE = re.compile(
    r"Ключевая\s*ставка[^<]*?</\w+>[^<]*?<[^>]*>\s*([\d]{1,2}[.,][\d]{1,2})\s*%",
    re.S | re.I,
)
_KEY_RATE_FALLBACK_RE = re.compile(
    r"Ключевая\s*ставка.{0,200}?(\d{1,2}[.,]\d{1,2})\s*%",
    re.S | re.I,
)


def get_key_rate(on_date: str | None = None) -> dict:
    d = _parse_date(on_date)

    if on_date is None or d == _date.today():
        try:
            html = _http_get(CBR_KEYIND_URL).decode("utf-8", errors="ignore")
            m = _KEY_RATE_RE.search(html) or _KEY_RATE_FALLBACK_RE.search(html)
            if m:
                val = float(m.group(1).replace(",", "."))
                return {"rate": val, "date": d.isoformat(), "source": "cbr_live"}
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass

    path = DATA_DIR / "key_rate_history.csv"
    chosen = None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rd = datetime.strptime(row["valid_from"], "%Y-%m-%d").date()
            if rd <= d:
                chosen = row
            else:
                break
    if chosen is None:
        return {"error": f"нет исторической ставки на {d}"}
    return {
        "rate": float(chosen["rate"]),
        "date": d.isoformat(),
        "valid_from": chosen["valid_from"],
        "source": "fallback_csv",
    }


# ===========================================================================
# 3. Инфляция (ИПЦ г/г, Росстат)
# ===========================================================================

def get_inflation(year: int, month: int) -> dict:
    year = int(year)
    month = int(month)
    if not (1 <= month <= 12):
        return {"error": f"month={month} вне 1..12"}

    path = DATA_DIR / "cpi_ru_monthly.csv"
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["year"]) == year and int(row["month"]) == month:
                return {
                    "year": year,
                    "month": month,
                    "cpi_yoy": float(row["cpi_yoy"]),
                    "source": "rosstat_csv",
                }
    return {"error": f"нет данных ИПЦ на {year}-{month:02d}"}


# ===========================================================================
# 4. Безработица (Росстат)
# ===========================================================================

def get_unemployment(year: int, month: int) -> dict:
    year = int(year)
    month = int(month)
    if not (1 <= month <= 12):
        return {"error": f"month={month} вне 1..12"}

    path = DATA_DIR / "unemployment_ru_monthly.csv"
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["year"]) == year and int(row["month"]) == month:
                return {
                    "year": year,
                    "month": month,
                    "unemployment": float(row["unemployment"]),
                    "source": "rosstat_csv",
                }
    return {"error": f"нет данных по безработице на {year}-{month:02d}"}


# ===========================================================================
# 5. Калькулятор
# ===========================================================================

def calculate(expression: str) -> dict:
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "пустое выражение"}

    allowed = set("0123456789.+-*/(),% ^")
    letters = re.findall(r"[A-Za-zА-Яа-я_]+", expression)
    blacklist = set(letters) - {
        "log", "ln", "sqrt", "exp", "pi", "E", "e",
        "sin", "cos", "tan", "abs",
    }
    if blacklist:
        return {"error": f"недопустимые идентификаторы: {sorted(blacklist)}"}

    try:
        val = float(sympy.sympify(expression.replace("^", "**")))
        return {"expression": expression, "result": round(val, 6)}
    except Exception as e:
        return {"expression": expression, "error": f"{type(e).__name__}: {e}"}


# ===========================================================================
# 6. compare_periods — ДЗ5, задание 1
# ===========================================================================

def compare_periods(
    metric: str,
    period_a: str,
    period_b: str,
) -> dict:
    """
    Сравнить значение метрики в двух периодах.

    Args:
        metric: "key_rate" | "fx_USD" | "fx_EUR" | "fx_CNY" | "cpi" | "unemployment"
        period_a: "YYYY-MM" или "YYYY-MM-DD"
        period_b: "YYYY-MM" или "YYYY-MM-DD"

    Returns:
        {"metric": ..., "a": {"date": ..., "value": ...},
         "b": {"date": ..., "value": ...},
         "delta": b.value - a.value,
         "ratio": b.value / a.value,
         "source": "..."}
    """

    def _get_value(m: str, period: str) -> tuple[float, str, str]:
        d = _parse_date(period)
        if m == "key_rate":
            r = get_key_rate(d.isoformat())
            if "error" in r:
                raise ValueError(r["error"])
            return r["rate"], r.get("date", d.isoformat()), r["source"]
        elif m.startswith("fx_"):
            currency = m[3:].upper()
            r = get_fx_rate(currency, d.isoformat())
            if "error" in r:
                raise ValueError(r["error"])
            return r["rate"], r.get("date", d.isoformat()), r["source"]
        elif m == "cpi":
            r = get_inflation(d.year, d.month)
            if "error" in r:
                raise ValueError(r["error"])
            return r["cpi_yoy"], f"{d.year}-{d.month:02d}", r["source"]
        elif m == "unemployment":
            r = get_unemployment(d.year, d.month)
            if "error" in r:
                raise ValueError(r["error"])
            return r["unemployment"], f"{d.year}-{d.month:02d}", r["source"]
        else:
            raise ValueError(
                f"Неизвестная метрика: {m!r}. "
                "Допустимые: key_rate, fx_USD, fx_EUR, fx_CNY, cpi, unemployment"
            )

    try:
        val_a, date_a, src_a = _get_value(metric, period_a)
        val_b, date_b, src_b = _get_value(metric, period_b)
    except ValueError as e:
        return {"error": str(e), "metric": metric, "period_a": period_a, "period_b": period_b}

    delta = round(val_b - val_a, 6)
    ratio = round(val_b / val_a, 6) if val_a != 0 else None

    return {
        "metric": metric,
        "a": {"date": date_a, "value": val_a},
        "b": {"date": date_b, "value": val_b},
        "delta": delta,
        "ratio": ratio,
        "source": f"{src_a}/{src_b}",
    }
