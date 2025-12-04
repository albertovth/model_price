import math

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="World Cup Ticket Toy Model", layout="centered")

st.title("World Cup Ticket Resale – Toy Model")

st.markdown(
    """
Modelá tu portafolio de boletos del Mundial como una mezcla de 3 tipos de partidos:

- **TOP**: partidos súper cotizados (final, semifinales, selecciones grandes, país anfitrión, etc.)
- **MID**: demanda normal, precios cerca del valor nominal
- **LOW**: poca demanda, posible venta por debajo de valor nominal
"""
)

# --- Sidebar: inputs ---------------------------------------------------------

st.sidebar.header("Supuestos")

n_tickets = st.sidebar.number_input(
    "Número de partidos para los que tenés boletos",
    min_value=1,
    max_value=104,
    value=24,
)

st.sidebar.subheader("Pesos (probabilidades relativas por tipo de partido)")
st.sidebar.markdown("No tienen que sumar 1, se van a normalizar automáticamente.")

w_top = st.sidebar.number_input("Peso TOP", min_value=0.0, value=0.10, step=0.01)
w_mid = st.sidebar.number_input("Peso MID", min_value=0.0, value=0.60, step=0.01)
w_low = st.sidebar.number_input("Peso LOW", min_value=0.0, value=0.30, step=0.01)

weights = np.array([w_top, w_mid, w_low], dtype=float)
if weights.sum() == 0:
    probs = np.array([0.0, 0.0, 0.0])
else:
    probs = weights / weights.sum()

st.sidebar.subheader("Multiplicadores de precio vs. valor nominal")
m_top = st.sidebar.number_input("TOP (por ejemplo 3.0 = 3× nominal)", min_value=0.0, value=3.0)
m_mid = st.sidebar.number_input("MID (por ejemplo 1.0 = igual a nominal)", min_value=0.0, value=1.0)
m_low = st.sidebar.number_input("LOW (por ejemplo 0.5 = 50 % de nominal)", min_value=0.0, value=0.5)

multipliers = np.array([m_top, m_mid, m_low], dtype=float)

# --- Calculations -----------------------------------------------------------

if probs.sum() == 0:
    st.error("Definí al menos un peso > 0 en la barra lateral.")
    st.stop()

expected_value = float(np.dot(probs, multipliers))  # EV por boleto

# Probabilidad de tener al menos un partido TOP
p_top = float(probs[0])
p_at_least_one_top = 1.0 - (1.0 - p_top) ** n_tickets

# --- Summary ----------------------------------------------------------------

st.header("Resumen matemático")

col1, col2 = st.columns(2)

with col1:
    st.write("**Probabilidades normalizadas por tipo de partido:**")
    st.write(f"TOP: **{probs[0]:.1%}**")
    st.write(f"MID: **{probs[1]:.1%}**")
    st.write(f"LOW: **{probs[2]:.1%}**")

with col2:
    st.write("**Multiplicadores de precio (vs. valor nominal):**")
    st.write(f"TOP: **{m_top:.2f}×**")
    st.write(f"MID: **{m_mid:.2f}×**")
    st.write(f"LOW: **{m_low:.2f}×**")

st.markdown(
    f"""
- **Valor esperado por boleto:** `EV ≈ {expected_value:.2f} × valor nominal`  
- **Probabilidad de que al menos uno de tus {n_tickets} partidos sea TOP:**  
  `P(≥1 TOP) ≈ {p_at_least_one_top:.1%}`
"""
)

# --- Monte Carlo simulation -------------------------------------------------

st.header("Simulación Monte Carlo del portafolio")

n_sims = st.slider(
    "Número de escenarios simulados",
    min_value=1_000,
    max_value=50_000,
    value=10_000,
    step=1_000,
)

rng = np.random.default_rng()
# 0 = TOP, 1 = MID, 2 = LOW
draw_indices = rng.choice(3, size=(n_sims, n_tickets), p=probs)
portfolio_values = multipliers[draw_indices].mean(axis=1)  # promedio × nominal

mean_sim = float(portfolio_values.mean())
q10, q25, q50, q75, q90 = np.quantile(portfolio_values, [0.1, 0.25, 0.5, 0.75, 0.9])

st.markdown(
    f"""
**Resultados de la simulación ({n_sims} escenarios):**

- Media del valor del portafolio: **{mean_sim:.2f} × nominal**
- Percentiles de la distribución del promedio por boleto:
  - 10 %: **{q10:.2f}×**
  - 25 %: **{q25:.2f}×**
  - 50 % (mediana): **{q50:.2f}×**
  - 75 %: **{q75:.2f}×**
  - 90 %: **{q90:.2f}×**
"""
)

fig, ax = plt.subplots()
ax.hist(portfolio_values, bins=40)
ax.set_xlabel("Valor promedio por boleto (× nominal)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución simulada del valor del portafolio")

st.pyplot(fig)

st.caption(
    "Jugá con los pesos (probabilidades) y los multiplicadores en la barra lateral para ver "
    "cómo va cambiando el riesgo: si la cola baja (LOW) es muy grande o muy barata, se vuelve más lotería."
)
