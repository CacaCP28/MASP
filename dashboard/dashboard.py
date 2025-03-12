# dashboard/dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import psycopg2

# Conecta ao banco usando a variável de ambiente
DATABASE_URL = os.getenv("postgresql://postgres:@gt2204@@db.cdlwtviryyppmwgizsau.supabase.co:5432/postgres")
conn = psycopg2.connect(DATABASE_URL)

query = "SELECT timestamp, count FROM contagem"
df = pd.read_sql(query, conn)
conn.close()

# Converter a coluna timestamp e criar colunas para data e hora
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['data'] = df['timestamp'].dt.date
df['hora'] = df['timestamp'].dt.hour

st.title("Dashboard de Contagem de Veículos")

st.subheader("Contagem Total")
total = df['count'].sum()
st.write(f"Total de veículos contados: {total}")

st.subheader("Contagem por Dia")
df_dia = df.groupby('data').sum().reset_index()
fig1, ax1 = plt.subplots()
ax1.bar(df_dia['data'].astype(str), df_dia['count'])
ax1.set_xlabel("Data")
ax1.set_ylabel("Quantidade")
ax1.set_title("Veículos por Dia")
st.pyplot(fig1)

st.subheader("Contagem por Hora")
df_hora = df.groupby('hora').sum().reset_index()
fig2, ax2 = plt.subplots()
ax2.bar(df_hora['hora'], df_hora['count'])
ax2.set_xlabel("Hora do Dia")
ax2.set_ylabel("Quantidade")
ax2.set_title("Veículos por Hora")
st.pyplot(fig2)
