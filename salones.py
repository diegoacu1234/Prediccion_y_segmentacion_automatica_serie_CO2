"""Lo que se sabe de cada salón (ip, id) además de las mediciones: horario, volumen y quiénes lo ocupan."""

from __future__ import annotations

from .mediciones import Group

# horario en que el salón está vacío, de lunes a viernes
HORARIOS = {("Escuelas32y83", 2): ("0:00-7:45", "10:00-10:30", "12:00-13:00", "15:00-15:30", "17:15-24:00")}

# volumen [m³]
VOLUMENES = {("Escuelas32y83", 2): 168.0, ("Escuelas32y83", 11): 168.0}

# Quiénes ocupan el salón: entre 20 y 30 escolares de 6 a 11 años y un docente. Tres casos de generación de CO2, de
# menor a mayor, con los valores del marco teórico de la tesis (Persily y de Jonge, 2017): metabolismo basal de
# 5,14 MJ/día para los escolares (31,9 kg) y de 8,24 MJ/día para el docente, que está a 1,6 met. El caso medio es el
# adoptado (25 escolares a 1,2 met); el bajo y el alto llevan a su extremo la cantidad (20 y 30) y la actividad
# (1,0 y 1,4 met) a la vez.
CLASE = {
    "bajo": Group(children=20, age=9, mass_kg=31.9, met=1.0, teacher_met=1.6, teacher_bmr=8.24, bmr=5.14),
    "medio": Group(children=25, age=9, mass_kg=31.9, met=1.2, teacher_met=1.6, teacher_bmr=8.24, bmr=5.14),
    "alto": Group(children=30, age=9, mass_kg=31.9, met=1.4, teacher_met=1.6, teacher_bmr=8.24, bmr=5.14),
}
CLASES = {("Escuelas32y83", 2): CLASE, ("Escuelas32y83", 11): CLASE}
