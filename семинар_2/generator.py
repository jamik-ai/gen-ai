import csv
import random
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_client import get_model, make_client
from schema import CITIES, Application

client = make_client()
MODEL = get_model()
N = 50

SPECIALITIES = [
    "врач", "учитель", "инженер", "бухгалтер", "юрист",
    "программист", "менеджер", "экономист", "психолог", "социальный работник",
]

SYSTEM_PROMPT = (
    "Ты генератор синтетических данных для системы ДПО. "
    "Создавай реалистичные заявки на курсы повышения квалификации от российских специалистов. "
    "Данные должны быть разнообразными и правдоподобными."
)


def make_prompt(seed_city: str, seed_speciality: str) -> str:
    return f"""Сгенерируй одну заявку на курс повышения квалификации для специалиста из города {seed_city}.

Поля:
- full_name: ФИО (реалистичное русское имя)
- age: целое число от 22 до 65
- address: объект с полями:
    city: ОБЯЗАТЕЛЬНО строка "{seed_city}" (не меняй)
    district: район или административный округ этого города
- speciality: ОБЯЗАТЕЛЬНО строка "{seed_speciality}" (не меняй)
- desired_course: одно из: "Управление проектами", "Цифровые технологии в профессии", "Финансовый менеджмент", "Педагогические технологии", "Охрана здоровья и медицина", "Правовое регулирование", "Психологическое консультирование", "Управление персоналом"
- years_of_experience: целое число от 0 до 40
- graduation_year: целое число от 1980 до 2024 (graduation_year + 22 ≤ 2026 + age)"""


def plot_histogram(data: list[str], title: str, filename: str) -> None:
    counts = Counter(data)
    labels = sorted(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("Количество заявок")
    plt.xticks(rotation=40, ha="right", fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранено: {filename}")


def main() -> None:
    # Стратификация: 10 городов × 5 специальностей = 50 пар (5 заявок на каждый город, 5 на каждую специальность)
    pairs = []
    for i, city in enumerate(CITIES[:10]):
        for j in range(5):
            speciality = SPECIALITIES[(i * 5 + j) % len(SPECIALITIES)]
            pairs.append((city, speciality))
    random.shuffle(pairs)

    applications: list[Application] = []
    errors = 0

    for i, (city, speciality) in enumerate(pairs):
        print(f"[{i + 1}/{N}] {city} / {speciality}...", end=" ", flush=True)
        try:
            app = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": make_prompt(city, speciality)},
                ],
                response_model=Application,
                max_retries=3,
                temperature=0.9,
            )
            applications.append(app)
            print(f"✓ {app.full_name}, {app.age}л, {app.speciality}")
        except Exception as e:
            errors += 1
            print(f"✗ {e}")
        time.sleep(0.2)

    print(f"\nРезультат: {len(applications)}/{N}, ошибок: {errors}")

    # CSV
    with open("applications.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "full_name", "age", "city", "district",
            "speciality", "desired_course",
            "years_of_experience", "graduation_year",
        ])
        writer.writeheader()
        for app in applications:
            writer.writerow({
                "full_name": app.full_name,
                "age": app.age,
                "city": app.address.city,
                "district": app.address.district,
                "speciality": app.speciality,
                "desired_course": app.desired_course,
                "years_of_experience": app.years_of_experience,
                "graduation_year": app.graduation_year,
            })
    print("Сохранено: applications.csv")

    # Гистограммы
    plot_histogram([a.address.city for a in applications], "Распределение по городам", "cities.png")
    plot_histogram([a.speciality for a in applications], "Распределение по специальностям", "specialities.png")

    # Статистика для выводов
    city_counts = Counter(a.address.city for a in applications)
    spec_counts = Counter(a.speciality for a in applications)
    n = len(applications)
    print("\nГорода:", {k: f"{v} ({100*v/n:.0f}%)" for k, v in city_counts.most_common()})
    print("Специальности:", {k: f"{v} ({100*v/n:.0f}%)" for k, v in spec_counts.most_common()})
    print(f"Макс. доля города: {max(city_counts.values())/n*100:.1f}% (порог 40%)")
    print(f"Макс. доля специальности: {max(spec_counts.values())/n*100:.1f}% (порог 35%)")


if __name__ == "__main__":
    main()
