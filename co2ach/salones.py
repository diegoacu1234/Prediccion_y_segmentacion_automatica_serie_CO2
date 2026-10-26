"""Lo que se sabe del salón A02 (Escuelas32y83, id 2) además de las mediciones: horario, volumen y clase."""

from __future__ import annotations

from .mediciones import Group

# horario en que el salón está vacío, de lunes a viernes
HORARIOS = {("Escuelas32y83", 2): ("0:00-7:45", "10:00-10:30", "12:00-13:00", "15:00-15:30", "17:15-24:00")}

# volumen [m³]
VOLUMENES = {("Escuelas32y83", 2): 168.0}

# Casos de generación (tabla 6.3): 20, 25 y 30 escolares a 1,0, 1,2 y 1,4 met, y un docente a 1,6 met.
# BMR de 5,14 MJ/día por escolar (31,9 kg) y 8,24 MJ/día el docente (apéndice A).
CLASE = {
    "bajo": Group(children=20, age=9, mass_kg=31.9, met=1.0, teacher_met=1.6, teacher_bmr=8.24, bmr=5.14),
    "medio": Group(children=25, age=9, mass_kg=31.9, met=1.2, teacher_met=1.6, teacher_bmr=8.24, bmr=5.14),
    "alto": Group(children=30, age=9, mass_kg=31.9, met=1.4, teacher_met=1.6, teacher_bmr=8.24, bmr=5.14),
}
CLASES = {("Escuelas32y83", 2): CLASE}
