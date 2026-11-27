
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ARCHIVO = "Proba3.csv"   


COL_PROMEDIO = "PROMEDIO"
COL_EDAD = "EDAD"
COL_GENERO = "GENERO"
COL_FACULTAD = "FACULTADES"
COL_SEMESTRE = "SEMESTRE"

EJE_PROMEDIO = (6, 10)
MIN_N = 3                      


ACTIVIDADES = ["Gimnasio y entrenamiento de fuerza", "Deportes", "Reuniones / salidas con amigos",
               "Salir a tomar un café", "Salidas de fiesta (antros / bares)",
               "Navegar en redes sociales e internet", "Jugar videojuegos", "Leer",
               "Actividades artísticas", "Tocar instrumentos", "Aprender nuevos hobbies",
               "Dormir", "Salir a caminar", "Conocer nuevos lugares de la ciudad",
               "Tiempo para ti"] #Guardamos todas las actividades en una lista

#Dividimos actividades en cada tipo
SOCIAL = ["Salidas de fiesta (antros / bares)", "Reuniones / salidas con amigos",
          "Salir a tomar un café"] 
FISICO = ["Gimnasio y entrenamiento de fuerza", "Deportes"]

ESCALA = {"1 o 2 veces al año": 1, "1 a 4 veces al semestre": 2, "1 a 3 veces al mes": 3,
          "1 a 2 veces por semana": 4, "Más de 2 veces por semana": 5}
ETIQ = {0: "No realiza", 1: "1-2/año", 2: "1-4/semestre", 3: "1-3/mes",
        4: "1-2/semana", 5: ">2/semana"}

#Identificamos cada semestre con un numero correspondiente
SEMESTRES = {"Primero": 1, "Segundo": 2, "Tercero": 3, "Cuarto": 4, "Quinto": 5,
             "Sexto": 6, "Séptimo": 7, "Septimo": 7, "Octavo": 8, "Noveno": 9, "Egresado": 10}
SEMESTRES.update({str(i): i for i in range(1, 10)})
ETIQ_SEM = {1: "1°", 2: "2°", 3: "3°", 4: "4°", 5: "5°", 6: "6°", 7: "7°", 8: "8°",
            9: "9°", 10: "Egresado"}

sns.set_theme(style="whitegrid")
os.makedirs("graficas", exist_ok=True)


#Se lee el archivo .csv columna a columna 
def buscar(df, texto, excluir):
    
    for c in df.columns:
        if texto.lower() in c.lower() and c not in excluir:
            return c
    raise KeyError(f"No encontré ninguna columna que contenga '{texto}'")


def ajustar(texto, ancho=18):
    return "\n".join(textwrap.wrap(str(texto), ancho))

#Definimos el eje de los promedios para cuando grafiquemos
def eje_promedio(ax, eje="y"):
    
    lo, hi = EJE_PROMEDIO
    marcas = range(int(np.ceil(lo)), hi + 1)
    if eje == "y":
        ax.set_ylim(lo, hi + 0.2)
        ax.set_yticks(marcas)
    else:
        ax.set_xlim(lo, hi + 0.2)
        ax.set_xticks(marcas)

#Se guardan las gráficas en una carpeta
def guardar(nombre):
    plt.tight_layout()
    plt.savefig(f"graficas/{nombre}.png", dpi=150, bbox_inches="tight")
    print("guardada:", nombre)

#Se define una funcion que crea boxplosts comparando el promedio con la categoria que querramos
def caja(ax, d, cat, orden, horizontal=False):
    
    n = d[cat].value_counts()
    d = d.assign(grupo=d[cat].map(lambda v: f"{v}\n(n={n[v]})"))
    orden = [f"{o}\n(n={n[o]})" for o in orden if o in n]
    if horizontal:
        sns.boxplot(data=d, x="promedio", y="grupo", order=orden, color="lightsteelblue",
                    showfliers=False, ax=ax)
        sns.stripplot(data=d, x="promedio", y="grupo", order=orden, color="0.3", size=3, ax=ax)
        eje_promedio(ax, "x")
    else:
        sns.boxplot(data=d, x="grupo", y="promedio", order=orden, color="lightsteelblue",
                    showfliers=False, ax=ax)
        sns.stripplot(data=d, x="grupo", y="promedio", order=orden, color="0.3", size=3, ax=ax)
        eje_promedio(ax, "y")
    ax.set(xlabel="" if not horizontal else "Promedio", ylabel="Promedio" if not horizontal else "")
    ax.tick_params(labelsize=8)



#Funcion que encuentra la correlación de espearman para cada categoria de actividad con el promedio
def spearman(x, y):
    
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 3 or d.iloc[:, 0].nunique() < 2:
        return np.nan, np.nan
    return stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])


#Se lee el archivo a través de pandas
raw = pd.read_csv(ARCHIVO, encoding="utf-8-sig")


#Definimos nombres más cortos y claves para llamar a cada columna de actividades
nombres = {a.lower(): a for a in ACTIVIDADES}
cols_act = {}                                  
for c in raw.columns:
    h = c.strip()
    if h.endswith("]") and "[" in h:
        h = h[h.rfind("[") + 1:-1].strip()
    if h.lower() in nombres:
        cols_act[nombres[h.lower()]] = c
if not cols_act:
    raise SystemExit("No se encontraron columnas de actividades. Revisa ACTIVIDADES.\n"
                     f"Encabezados de tu CSV: {list(raw.columns)}")
faltan = [a for a in ACTIVIDADES if a not in cols_act]
if faltan:
    print(f"AVISO: se encontrarpm estas actividades en el CSV: {faltan}")
usadas = list(cols_act.values())
cols = {}
for nombre, texto in [("promedio", COL_PROMEDIO), ("edad", COL_EDAD), ("genero", COL_GENERO),
                      ("facultad", COL_FACULTAD), ("semestre", COL_SEMESTRE)]:
    cols[nombre] = buscar(raw, texto, usadas)
    usadas.append(cols[nombre])
df = pd.DataFrame({nombre: raw[c] for nombre, c in cols.items()})

#Pone restricción a promedios para no incluir los que pongan arriba de 10
df["promedio"] = pd.to_numeric(df["promedio"].astype(str).str.replace(",", "."), errors="coerce")
fuera = df["promedio"].notna() & ~df["promedio"].between(0, 10)
if fuera.any():
    print(f"AVISO: {fuera.sum()} promedios fuera de 0-10 se excluyen: {df.loc[fuera, 'promedio'].tolist()}")
df["promedio"] = df["promedio"].where(df["promedio"].between(0, 10)).round(2)

df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
df["semestre"] = df["semestre"].astype(str).str.strip().str.capitalize().map(SEMESTRES)


acts = []
for nombre, c in cols_act.items():
    valores = raw[c].fillna("0").astype(str).str.strip()  
    raros = set(valores) - set(ESCALA) - {"0"}
    if raros:
        print(f"AVISO: valores no reconocidos en '{nombre}': {raros} (se toman como 0)")
    df[nombre] = valores.map(ESCALA).fillna(0)
    acts.append(nombre)


df = df.dropna(subset=["promedio"]).reset_index(drop=True)
print(f"{len(df)} respuestas válidas, {len(acts)} actividades\n")


facs = df["facultad"].value_counts()
facs = facs[facs >= MIN_N].index.tolist()
df["sem_txt"] = df["semestre"].map(ETIQ_SEM)


#Descripción de la muestra
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
sns.histplot(df["promedio"], bins=np.arange(0, 10.5, 0.5), color="steelblue", ax=ax[0, 0])
ax[0, 0].axvline(df["promedio"].mean(), color="red", ls="--",
                 label=f"Media = {df['promedio'].mean():.2f}")
ax[0, 0].axvline(df["promedio"].median(), color="green", ls="-.",
                 label=f"Mediana = {df['promedio'].median():.2f}")
ax[0, 0].set(title="Distribución del promedio", xlabel="Promedio", ylabel="Frecuencia")
ax[0, 0].legend()
eje_promedio(ax[0, 0], "x")
sns.histplot(df["edad"].dropna(), discrete=True, color="darkorange", ax=ax[0, 1])
ax[0, 1].set(title="Distribución de la edad", xlabel="Edad", ylabel="Frecuencia")
sem = df["sem_txt"].value_counts().reindex([ETIQ_SEM[i] for i in ETIQ_SEM]).dropna()
sem.plot.bar(color="slateblue", ax=ax[0, 2], rot=0)
ax[0, 2].set(title="Estudiantes por semestre", xlabel="", ylabel="Cantidad")
df["genero"].value_counts().plot.bar(color="mediumpurple", ax=ax[1, 0], rot=0)
ax[1, 0].set(title="Estudiantes por género", xlabel="", ylabel="Cantidad")
df["facultad"].value_counts().sort_values().plot.barh(color="seagreen", ax=ax[1, 1])
ax[1, 1].set(title="Estudiantes por facultad", xlabel="Cantidad", ylabel="")
ax[1, 2].axis("off")
guardar("01_descripcion_muestra")

# Se crean las gráficas de promedio vs. género, facultad, semestre y edad
fig, ax = plt.subplots(2, 2, figsize=(16, 11))
caja(ax[0, 0], df, "genero", df["genero"].value_counts().index)
ax[0, 0].set_title("Promedio por género")
d = df[df["facultad"].isin(facs)]
caja(ax[0, 1], d, "facultad", d.groupby("facultad")["promedio"].median().sort_values().index,
     horizontal=True)
ax[0, 1].set_title(f"Promedio por facultad (n ≥ {MIN_N})")
caja(ax[1, 0], df.dropna(subset=["sem_txt"]), "sem_txt", list(ETIQ_SEM.values()))
ax[1, 0].set_title("Promedio por semestre")
sns.scatterplot(data=df, x="edad", y="promedio", hue="genero", s=60, ax=ax[1, 1])
rho, p = spearman(df["edad"], df["promedio"])
ax[1, 1].set(title=f"Edad vs. promedio (Spearman ρ = {rho:.2f}, p = {p:.3f})",
             xlabel="Edad", ylabel="Promedio")
eje_promedio(ax[1, 1], "y")
guardar("02_promedio_vs_demograficas")




#Promedio a partir de la frecuencia de cada actividad
filas = int(np.ceil(len(acts) / 3))
fig, ax = plt.subplots(filas, 3, figsize=(17, 4.2 * filas))
ax = ax.ravel()
for a, eje in zip(acts, ax):
    d = pd.DataFrame({"frecuencia": df[a].map(ETIQ), "promedio": df["promedio"]})
    caja(eje, d, "frecuencia", list(ETIQ.values()))
    rho, p = spearman(df[a], df["promedio"])
    eje.set_title(f"{ajustar(a, 34)}\nρ = {rho:.2f}", fontsize=10)
for eje in ax[len(acts):]:
    eje.axis("off")
guardar("03_boxplots_promedio_por_actividad")


# Se calcula la correlación de Spearman de cada actividad con el promedio
tabla = pd.DataFrame([(a, *spearman(df[a], df["promedio"])) for a in acts],
                     columns=["actividad", "rho", "p_valor"]).dropna().sort_values("rho")
plt.figure(figsize=(10, 0.5 * len(tabla) + 2))
plt.barh(tabla["actividad"], tabla["rho"],
         color=["seagreen" if r > 0 else "indianred" for r in tabla["rho"]])
for i, (r, p) in enumerate(zip(tabla["rho"], tabla["p_valor"])):
    plt.text(r, i, f" {r:.2f}{'*' if p < 0.05 else ''} ", va="center",
             ha="left" if r >= 0 else "right", fontsize=9)
plt.axvline(0, color="black", lw=0.8)
plt.xlim(-0.6, 0.6)
plt.xlabel("Correlación de Spearman con el promedio (* = p < 0.05)")
plt.title("¿Qué actividades se relacionan con el promedio?")
guardar("04_spearman_actividades_vs_promedio")
tabla.sort_values("rho", ascending=False).to_csv("graficas/spearman_actividades.csv", index=False)


# Heatmap de correlaciones entre actividades, semestre y promedio
cols_corr = acts + ["semestre", "promedio"]
corr = df[cols_corr].corr(method="spearman")
plt.figure(figsize=(13, 11))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), k=1), annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1, square=True, annot_kws={"size": 8})
plt.xticks(np.arange(len(cols_corr)) + 0.5, [ajustar(c, 16) for c in cols_corr],
           rotation=60, ha="right", fontsize=8)
plt.yticks(np.arange(len(cols_corr)) + 0.5, [ajustar(c, 24) for c in cols_corr], fontsize=8)
plt.title("Correlación de Spearman")
guardar("05_heatmap_correlacion")


#Frecuencia media de cada actividad por facultad y por semestre
for grupo, datos, nombre in [("facultad", df[df["facultad"].isin(facs)], "06a_actividad_por_facultad"),
                             ("sem_txt", df.dropna(subset=["sem_txt"]), "06b_actividad_por_semestre")]:
    medias = datos.groupby(grupo)[acts].mean()
    if grupo == "sem_txt":
        medias = medias.reindex([e for e in ETIQ_SEM.values() if e in medias.index])
    n = datos[grupo].value_counts()
    medias.index = [f"{g} (n={n[g]})" for g in medias.index]
    plt.figure(figsize=(14, 0.6 * len(medias) + 4))
    sns.heatmap(medias, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=5,
                cbar_kws={"label": "Frecuencia media (0-5)"})
    plt.xticks(np.arange(len(acts)) + 0.5, [ajustar(a, 16) for a in acts],
               rotation=60, ha="right", fontsize=8)
    plt.yticks(rotation=0)
    plt.title("Frecuencia media de cada actividad por " + ("facultad" if grupo == "facultad" else "semestre"))
    guardar(nombre)

# Índices de ocio social y ocio físico
df["ocio_social"] = df[[a for a in SOCIAL if a in acts]].sum(axis=1)
df["ocio_fisico"] = df[[a for a in FISICO if a in acts]].sum(axis=1)
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
for eje, (x, y) in zip(ax, [("ocio_social", "promedio"), ("ocio_fisico", "promedio"),
                            ("ocio_social", "ocio_fisico")]):
    eje.scatter(df[x] + np.random.normal(0, 0.1, len(df)), df[y], alpha=0.7, color="steelblue")
    if df[x].nunique() > 1:
        m, b = np.polyfit(df[x], df[y], 1)
        xs = np.array([df[x].min(), df[x].max()])
        eje.plot(xs, m * xs + b, color="red", ls="--")
    rho, p = spearman(df[x], df[y])
    eje.set(title=f"{x.replace('_', ' ')} vs. {y.replace('_', ' ')}\nρ = {rho:.2f} (p = {p:.3f})",
            xlabel=x.replace("_", " ") + " (suma de escalas 0-5)",
            ylabel="Promedio" if y == "promedio" else y.replace("_", " "))
    if y == "promedio":
        eje_promedio(eje, "y")
guardar("07_indices_social_fisico")


print("\nCorrelación de Spearman con el promedio")
for nombre in ["edad", "semestre"]:
    rho, p = spearman(df[nombre], df["promedio"])
    print(f"  {nombre}: rho = {rho:.2f}, p = {p:.3f}")
print("\nActividades con p < 0.05:")
signif = tabla[tabla["p_valor"] < 0.05]
print(signif.to_string(index=False) if len(signif) else "  (ninguna)")

print("\nKruskal-Wallis: ¿el promedio cambia entre grupos?")
for grupo in ["genero", "facultad", "semestre"]:
    datos = df[df["facultad"].isin(facs)] if grupo == "facultad" else df
    grupos = [g["promedio"].values for _, g in datos.groupby(grupo) if len(g) >= 2]
    if len(grupos) >= 2:
        h, p = stats.kruskal(*grupos)
        print(f"  {grupo}: H = {h:.2f}, p = {p:.3f}")

print("\nCon muchas pruebas a la vez, alguna saldrá significativa por azar.")



# MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN 
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

#Definimos una funcion para las medidas
def medidas(serie):
    
    serie = serie.dropna()
    cuenta = serie.value_counts()
    if cuenta.empty or cuenta.max() == 1:
        moda = "sin moda"                       # todos los valores son distintos
    else:
        moda = ", ".join(str(round(float(m), 2)) for m in serie.mode()[:3])
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    media, desv = serie.mean(), serie.std()
    return {
        "n": len(serie),
        # tendencia central
        "media": round(media, 2),
        "mediana": round(serie.median(), 2),
        "moda": moda,
        # dispersión
        "desv. est.": round(desv, 2),
        "varianza": round(serie.var(), 2),
        "mín": round(serie.min(), 2),
        "máx": round(serie.max(), 2),
        "rango": round(serie.max() - serie.min(), 2),
        "Q1": round(q1, 2),
        "Q3": round(q3, 2),
        "IQR": round(q3 - q1, 2),
        "CV (%)": round(desv / media * 100, 1) if media != 0 else np.nan,
    }


print("\n" + "=" * 90)
print("MEDIDAS DESCRIPTIVAS: PROMEDIO Y EDAD")
print("=" * 90)
print(pd.DataFrame({"promedio": medidas(df["promedio"]), "edad": medidas(df["edad"])}).to_string())


print("\n" + "=" * 90)
print("MEDIDAS DESCRIPTIVAS: FRECUENCIA DE CADA ACTIVIDAD (escala 0-5)")
print("=" * 90)
print(pd.DataFrame({a: medidas(df[a]) for a in acts}).T.to_string())
print("Nota: la escala es ordinal; para ella son más apropiadas la mediana, la moda y el IQR.")

for col, titulo in [("genero", "GÉNERO"), ("facultad", "FACULTAD"), ("sem_txt", "SEMESTRE")]:
    datos = df[df["facultad"].isin(facs)] if col == "facultad" else df
    por_grupo = datos.groupby(col)["promedio"].agg(
        n="count", media="mean", mediana="median", desv_est="std",
        mín="min", máx="max", Q1=lambda s: s.quantile(0.25), Q3=lambda s: s.quantile(0.75))
    por_grupo["IQR"] = por_grupo["Q3"] - por_grupo["Q1"]
    if col == "sem_txt":                        # orden natural de los semestres
        por_grupo = por_grupo.reindex([e for e in ETIQ_SEM.values() if e in por_grupo.index])
    print("\n" + "=" * 90)
    print(f"PROMEDIO POR {titulo}")
    print("=" * 90)
    print(por_grupo.round(2).to_string())



plt.show()