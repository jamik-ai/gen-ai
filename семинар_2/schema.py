from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CURRENT_YEAR = date.today().year

CITIES = [
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург",
    "Казань", "Нижний Новгород", "Самара", "Краснодар",
    "Ростов-на-Дону", "Уфа", "Омск", "Красноярск",
]


class Address(BaseModel):
    city: str
    district: str

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v


class Application(BaseModel):
    full_name: str
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Literal[
        "врач", "учитель", "инженер", "бухгалтер",
        "юрист", "программист", "менеджер", "экономист",
        "психолог", "социальный работник",
    ]
    desired_course: Literal[
        "Управление проектами",
        "Цифровые технологии в профессии",
        "Финансовый менеджмент",
        "Педагогические технологии",
        "Охрана здоровья и медицина",
        "Правовое регулирование",
        "Психологическое консультирование",
        "Управление персоналом",
    ]
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @field_validator("graduation_year")
    @classmethod
    def graduation_not_in_future(cls, v: int) -> int:
        if v > CURRENT_YEAR:
            raise ValueError(f"Год окончания {v} не может быть в будущем (сейчас {CURRENT_YEAR})")
        return v

    @model_validator(mode="after")
    def age_and_graduation_consistent(self) -> "Application":
        # graduation_year + 22 ≤ current_year + age: возраст и год окончания не противоречат друг другу
        if self.graduation_year + 22 > CURRENT_YEAR + self.age:
            raise ValueError(
                f"Год окончания {self.graduation_year} противоречит возрасту {self.age}: "
                f"при выпуске в {self.graduation_year} человеку было бы "
                f"{self.graduation_year - (CURRENT_YEAR - self.age)} лет"
            )
        return self
