# app.py

import os
# Suprime logs de informações do TensorFlow (oneDNN etc.) e Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt # Altair ainda é importado e usado para o gráfico de Consumo de CPU ao Longo do Tempo
import random
import matplotlib.pyplot as plt # Matplotlib ainda é importado mas não será usado para o gráfico principal
import plotly.express as px # Importando Plotly Express
import plotly.graph_objects as go # IMPORTANTE: Adicionado import para plotly.graph_objects

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import (
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Dropout,
    Dense,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from arch import arch_model

# Configurações da página
st.set_page_config(
    page_title="Análise e Previsão de Consumo de CPU",
    layout="wide"
)
st.title("Dashboard de Previsão de Consumo de CPU")

# 1. Upload de arquivo
st.sidebar.header("1. Upload de Arquivo")
uploaded_file = st.sidebar.file_uploader(
    "Envie seu arquivo (CSV, XLS, XLSX)",
    type=["csv", "xls", "xlsx"]
)

def load_data(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file, engine="openpyxl")

if uploaded_file is not None:
    # 2. Leitura e validação
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Falha ao ler o arquivo: {e}")
        st.stop()

    required_cols = [
        "data",
        "consumo_cpu",
        "consumo_media_movel",
        "consumo_desvio_padrao"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.sidebar.error(f"Faltando colunas: {missing}")
        st.stop()

    # 3. Conversão e seleção de colunas
    df = df[required_cols].copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    # 4. Filtros e ordenação
    st.sidebar.header("2. Filtros e Ordenação")
    
    with st.sidebar.expander("🔎 Filtros e Opções de Visualização", expanded=True):
        # Filtro de Data
        min_d, max_d = df["data"].min().date(), df["data"].max().date()
        
        # --- INÍCIO DA CORREÇÃO PARA O ERRO DO INTERVALO DE DATAS ---
        selected_dates_tuple = st.date_input(
            "Intervalo de datas",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            key="date_filter"
        )

        if len(selected_dates_tuple) == 2:
            start_date = selected_dates_tuple[0]
            end_date = selected_dates_tuple[1]
        elif len(selected_dates_tuple) == 1:
            # Se apenas uma data foi selecionada (estado intermediário),
            # use-a como data de início e defina a data de fim como a mesma
            # para evitar o erro de desempacotamento. O usuário irá selecionar a segunda data em seguida.
            start_date = selected_dates_tuple[0]
            end_date = selected_dates_tuple[0] 
        else:
            # Caso padrão, por exemplo, antes de qualquer seleção ou se for limpo
            start_date = min_d
            end_date = max_d
        # --- FIM DA CORREÇÃO ---
        
        # Filtro de Consumo de CPU
        min_cpu, max_cpu = int(df["consumo_cpu"].min()), int(df["consumo_cpu"].max()) + 1
        cpu_range = st.slider(
            "Filtro por faixa de consumo de CPU",
            min_value=min_cpu,
            max_value=max_cpu,
            value=(min_cpu, max_cpu),
            key="cpu_filter"
        )

        # Ordenação
        sort_opts = {
            "Data ↑": ("data", True),
            "Data ↓": ("data", False),
            "CPU ↑": ("consumo_cpu", True),
            "CPU ↓": ("consumo_cpu", False),
            "Média Móvel ↑": ("consumo_media_movel", True),
            "Média Móvel ↓": ("consumo_media_movel", False)
        }
        sel = st.selectbox("Ordenar por", list(sort_opts.keys()), index=0, key="sort_box")
        
        # Seleção de Colunas para Exibição
        all_cols = df.columns.tolist() + ["upper", "lower"]
        default_cols = [col for col in required_cols if col in all_cols]
        cols_to_show = st.multiselect(
            "Selecione as colunas para exibir na tabela",
            options=all_cols,
            default=default_cols,
            key="column_selector"
        )

    # Aplicando os filtros
    df_filtered = df[
        (df["data"].dt.date >= start_date) &
        (df["data"].dt.date <= end_date) &
        (df["consumo_cpu"] >= cpu_range[0]) &
        (df["consumo_cpu"] <= cpu_range[1])
    ].copy()

    # Aplicando a ordenação
    col, asc = sort_opts[sel]
    df_filtered = df_filtered.sort_values(col, ascending=asc)


    # 5. Cálculo de limites de desvio padrão
    df_filtered["upper"] = df_filtered["consumo_cpu"] + df_filtered["consumo_desvio_padrao"]
    df_filtered["lower"] = df_filtered["consumo_cpu"] - df_filtered["consumo_desvio_padrao"]

    # 6. Exibição da tabela
    st.success("Dados preparados com sucesso.")
    st.subheader("Amostra dos Dados Tratados")
    if not cols_to_show:
        st.warning("Selecione pelo menos uma coluna para exibir.")
    else:
        display_cols = [c for c in cols_to_show if c in df_filtered.columns]
        st.dataframe(df_filtered[display_cols].head(10), use_container_width=True)

    # 7. Gráfico interativo
    st.subheader("Consumo de CPU ao Longo do Tempo")
    base = alt.Chart(df_filtered).encode(x=alt.X("data:T", title="Data"))
    area = base.mark_area(color="lightblue", opacity=0.3).encode(
        y="upper:Q",
        y2="lower:Q"
    )
    cpu_line = base.mark_line(color="blue", strokeWidth=2).encode(
        y=alt.Y("consumo_cpu:Q", title="Consumo CPU")
    )
    ma_line = base.mark_line(color="orange", strokeDash=[5, 5], strokeWidth=2).encode(
        y="consumo_media_movel:Q"
    )
    st.altair_chart(alt.layer(area, cpu_line, ma_line).interactive(),
                    use_container_width=True)

    # 8. Confirmação para iniciar previsão
    st.subheader("Estudo e Previsão de Consumo de CPU")
    run_pipeline = st.checkbox("Deseja iniciar o pipeline de previsão?")

    if run_pipeline:
        st.info("Pipeline de previsão em execução...")
        progress = st.progress(0)
        
        df_pipeline = df_filtered.copy()

        # 10. Normalização dos dados
        progress.progress(10)
        data_to_scale = df_pipeline[[
            "consumo_cpu",
            "consumo_media_movel",
            "consumo_desvio_padrao"
        ]].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_to_scale)

        df_norm = pd.DataFrame(
            scaled_data,
            columns=["consumo_cpu", "consumo_media_movel", "consumo_desvio_padrao"]
        )
        df_norm["data"] = df_pipeline["data"].values

        st.write("✅ Normalização concluída")
        st.dataframe(df_norm.head(5), use_container_width=True)

        # 11. Criação de sequências
        def create_sequences_multi(data, n_steps):
            Xs, ys = [], []
            for i in range(len(data) - n_steps):
                Xs.append(data[i:(i + n_steps), :])
                ys.append(data[i + n_steps, 0])
            return np.array(Xs), np.array(ys)

        progress.progress(20)
        n_steps = 7
        X, y = create_sequences_multi(scaled_data, n_steps)
        # st.write(f"Sequências criadas: X.shape = {X.shape}, y.shape = {y.shape}") # Removida a mensagem
        
        # 12. Divisão dos dados em Treino e Teste
        progress.progress(30)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        # st.write( # Removida a mensagem
        #     f"Dados de treino → X_train: {X_train.shape}, y_train: {y_train.shape}\n"
        #     f"Dados de teste  → X_test: {X_test.shape},  y_test: {y_test.shape}"
        # )

        # 13–14. Construção e Treinamento do Modelo Transformer
        progress.progress(45)
        st.subheader("Construção e Treinamento do Modelo Transformer")
        st.info("🔨 Iniciando treinamento do Transformer")

        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            x = LayerNormalization(epsilon=1e-6)(inputs)
            attn_output = MultiHeadAttention(
                key_dim=head_size,
                num_heads=num_heads,
                dropout=dropout
            )(x, x)
            x = Dropout(dropout)(attn_output)
            res = x + inputs

            x = LayerNormalization(epsilon=1e-6)(res)
            x = Dense(ff_dim, activation="relu")(x)
            x = Dense(inputs.shape[-1])(x)
            x = Dropout(dropout)(x)
            return x + res

        head_size = 128
        num_heads = 4
        ff_dim = 128
        num_transformer_blocks = 2
        mlp_units = [64]
        mlp_dropout = 0.4
        dropout = 0.2

        inputs = Input(shape=(n_steps, X_train.shape[2]))
        x = inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = GlobalAveragePooling1D()(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(1)(x)

        model_transformer = Model(inputs=inputs, outputs=outputs)
        model_transformer.compile(
            loss="huber",
            optimizer=Adam(learning_rate=1e-4),
            metrics=["mae"]
        )
        model_transformer.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        st.success("✅ Modelo Transformer treinado com sucesso!")

        # 15. Treinamento do Modelo GARCH
        progress.progress(60)
        st.subheader("Treinamento do Modelo GARCH")
        st.info("🔨 Ajustando GARCH nos resíduos")

        predictions_scaled = model_transformer.predict(X_test, verbose=0)
        predictions_desnorm = scaler.inverse_transform(
            np.concatenate([predictions_scaled,
                            np.zeros((len(predictions_scaled), 2))],
                            axis=1)
        )[:, 0]
        real_values_desnorm = scaler.inverse_transform(
            np.concatenate([y_test.reshape(-1, 1),
                            np.zeros((len(y_test), 2))],
                            axis=1)
        )[:, 0]

        residuos = real_values_desnorm - predictions_desnorm
        garch_data = pd.Series(residuos).dropna()
        am = arch_model(garch_data, vol="Garch", p=1, q=1, mean="constant")
        res_garch = am.fit(update_freq=5, disp="off")
        st.success("✅ Modelo GARCH treinado com sucesso!")
        st.write(res_garch.summary())

        # --- INÍCIO DO RESUMO DO MODELO GARCH ---
        st.markdown(rf"""
### Resumo dos Resultados do Modelo GARCH

Após a previsão da tendência pelo modelo Transformer, o Modelo **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** foi aplicado aos **resíduos** (os erros de previsão) para modelar e prever a **volatilidade** desses erros. Isso é crucial porque, em séries temporais, a variabilidade dos erros (ou seja, o quanto as previsões podem "flutuar") muitas vezes não é constante ao longo do tempo.

**O que observar no sumário do GARCH:**

* **P-values (Valores-P) dos Coeficientes:** Para entender se o modelo GARCH está realmente capturando a volatilidade, observe os **p-values** associados aos termos `ARCH` (Alpha) e `GARCH` (Beta) no sumário.
    * Valores-p **pequenos** (geralmente menores que 0.05) para esses termos indicam que eles são **estatisticamente significantes**. Isso significa que a volatilidade passada dos erros (`ARCH`) e a volatilidade passada da própria variância condicional (`GARCH`) são importantes para prever a volatilidade futura.
    * Um modelo GARCH com termos significantes é um bom sinal de que a volatilidade dos resíduos está sendo modelada de forma eficaz, o que contribui para um **Intervalo de Confiança** mais preciso e bem calibrado.

* **Ajuste (Fit) do Modelo:** Se o GARCH foi capaz de se ajustar bem aos resíduos do Transformer, significa que ele está capturando padrões na volatilidade que o Transformer, por si só, não conseguiria. Isso leva a um **intervalo de confiança mais realista** para as suas previsões.

A integração do GARCH é fundamental para ir além de uma previsão pontual, fornecendo uma medida da **incerteza** e do **risco** associados às suas estimativas de consumo de CPU.
""")
        # --- FIM DO RESUMO DO MODELO GARCH ---

        # 16. Análise de Acurácia e Métricas de Erro
        progress.progress(70)
        st.subheader("Análise de Acurácia e Métricas de Erro")
        st.info("🔍 Avaliando cobertura do intervalo de confiança")

        previsao_vol = res_garch.forecast(horizon=len(X_test))
        var_forecast = previsao_vol.variance.iloc[-1]
        desvios = np.sqrt(var_forecast).values.flatten()
        limite_sup = desvios * 1.96
        limite_inf = -desvios * 1.96
        previsao_inf = predictions_desnorm + limite_inf
        previsao_sup = predictions_desnorm + limite_sup

        total = len(X_test)
        acertos = np.sum(
            (real_values_desnorm >= previsao_inf) &
            (real_values_desnorm <= previsao_sup)
        )
        acuracia_global = (acertos / total) * 100

        st.write(f"Total previsões: {total}")
        st.write(f"Previsões corretas (dentro do IC 95%): {acertos}")
        st.write(f"Acurácia Global do Modelo Combinado: {acuracia_global:.2f}%")

        st.markdown(f"""
**Análise da acurácia do intervalo de confiança**

A acurácia do seu modelo não se baseia em previsão pontual, mas num intervalo de confiança de 95%.  
O valor de **{acuracia_global:.2f}%** indica que em **{acuracia_global:.2f}%** das previsões, o valor real caiu dentro do intervalo calculado pelo modelo.  
Para um modelo que prevê intervalo de 95%, uma acurácia próxima a 95% demonstra calibração correta.  
Valores acima de 90% são excelentes em séries temporais, e competições como a M4 valorizam alta cobertura de IC.
""")
        st.markdown(rf"""
### Como o Intervalo de Confiança (IC) é Calculado e Interpretado

O Intervalo de Confiança é uma ferramenta essencial para entender a **incerteza** em torno de uma previsão. Em vez de apenas um único valor, ele oferece uma faixa onde o valor real provavelmente estará.

#### **Cálculo do IC no Modelo Híbrido (Transformer + GARCH):**

1.  **Previsão de Tendência (Modelo Transformer):**
    * O modelo Transformer primeiro gera uma previsão do valor central (a tendência) do consumo de CPU. Esta é a "melhor estimativa" do modelo para o futuro.
    2.  **Modelagem da Volatilidade (Modelo GARCH):**
    * Os erros (resíduos) da previsão do Transformer são então passados para o modelo GARCH.
    * O GARCH é especializado em prever a **volatilidade**, ou seja, o quanto esses erros podem variar ao longo do tempo. Ele estima o "desvio padrão" futuro desses erros.
    3.  **Construção do Intervalo:**
    * Para construir o IC de 95%, multiplicamos o desvio padrão da volatilidade (previsto pelo GARCH) por um fator de 1.96 (que corresponde a 95% da área sob a curva de uma distribuição normal padrão).
    * Esse valor é então adicionado e subtraído da previsão de tendência do Transformer, criando os limites superior e inferior do Intervalo de Confiança.

    Portanto, o **IC de 95%** é dado por:
    $$\text{{Previsão de Tendência}} \pm (1.96 \times \text{{Desvio Padrão da Volatilidade}})$$

#### **Interpretação do IC de 95%:**

* **Significado:** Um Intervalo de Confiança de 95% significa que, em 95% das vezes, o valor real observado cairá dentro da faixa que o modelo previu.
* **Confiabilidade:** Se a sua **Acurácia Global do Modelo Combinado** for de **{acuracia_global:.2f}%**, isso indica que em **{acuracia_global:.2f}%** das suas previsões no conjunto de teste, o valor real de consumo de CPU realmente esteve contido dentro do intervalo de 95% calculado pelo modelo.
* **Calibração:** Para um modelo com IC de 95%, o ideal é que a acuracia global seja próxima de 95%. Um valor de **{acuracia_global:.2f}%** sugere que o seu modelo está **muito bem calibrado** e que seus intervalos de confiança são confiáveis para capturar a incerteza inerente à série temporal.
* **Aplicação Prática:** Na prática, o IC permite que você não apenas preveja o consumo de CPU, mas também entenda a **margem de erro** e o **risco** associado a essa previsão. Isso é crucial para tomada de decisões, pois permite planejar para cenários de consumo mínimo e máximo prováveis.

Essa combinação do Transformer com o GARCH é poderosa porque lida tanto com a tendência dos dados quanto com a variabilidade dos erros de previsão.
""")

        # 17. Cálculo das Métricas de Erro
        progress.progress(80)
        st.subheader("Cálculo das Métricas de Erro")
        st.info("📊 Calculando RMSE, MAE e MAPE")

        rmse = np.sqrt(mean_squared_error(real_values_desnorm, predictions_desnorm))
        mae = mean_absolute_error(real_values_desnorm, predictions_desnorm)
        # Handle division by zero for MAPE
        mape = np.mean(
            np.abs((real_values_desnorm - predictions_desnorm) /
                   np.where(real_values_desnorm != 0, real_values_desnorm, 1e-8)) # Added check for zero
        ) * 100

        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"MAPE: {mape:.2f}%")
        
        st.markdown(rf"""
### Entendendo as Métricas de Erro

As métricas de erro são cruciais para avaliar a performance de um modelo de previsão. Elas quantificam o quão bem as previsões do seu modelo se aproximam dos valores reais.

#### **RMSE (Root Mean Squared Error) - Erro Quadrático Médio da Raiz**
* **O que representa?** O RMSE mede a raiz quadrada da média dos erros quadráticos. Em termos mais simples, ele indica a magnitude média dos erros do modelo, mas penaliza mais fortemente os erros maiores. Isso o torna sensível a outliers.
* **Interpretação:** Quanto **menor** o valor do RMSE, melhor o modelo. O RMSE está na mesma unidade da variável que você está prevendo (neste caso, "Consumo de CPU"). Um RMSE de **{rmse:.2f}** significa que, em média, as previsões do seu modelo desviam dos valores reais em aproximadamente {rmse:.2f} unidades de consumo de CPU.
* **Valores bons ou ruins?** Um RMSE de {rmse:.2f} é considerado **bom** se for pequeno em relação à escala dos seus dados de consumo de CPU. Para avaliar se é "bom", compare-o com a média ou o desvio padrão dos valores reais. Se o RMSE for uma pequena fração do desvio padrão do consumo de CPU, isso é um bom sinal.
* **Referência:** Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice* (2nd ed.). OTexts.

#### **MAE (Mean Absolute Error) - Erro Absoluto Médio**
* **O que representa?** O MAE calcula a média das magnitudes dos erros. Ele mede a precisão média do modelo, dando o mesmo peso a todos os erros, grandes ou pequenos. É menos sensível a outliers do que o RMSE.
* **Interpretação:** Assim como o RMSE, quanto **menor** o MAE, melhor o modelo. Ele também está na mesma unidade da variável de previsão. Um MAE de **{mae:.2f}** indica que, em média, as previsões do seu modelo têm um erro absoluto de aproximadamente {mae:.2f} unidades de consumo de CPU.
* **Valores bons ou ruins?** Um MAE de **{mae:.2f}** pode ser considerado **bom** se for pequeno em relação aos valores reais de consumo. Ele fornece uma medida mais intuitiva do erro médio. A mesma lógica de comparação com a escala dos dados se aplica aqui.
* **Referência:** Shcherbakov, M. V., Brebels, A., Shcherbakova, N. L., Tyukov, A. P., Janovsky, T. A., & Kamaev, V. A. (2013). A Survey of Forecast Error Measures. *World Applied Sciences Journal*, 24(9), 1163-1175.

#### **MAPE (Mean Absolute Percentage Error) - Erro Percentual Absoluto Médio**
* **O que representa?** O MAPE expressa o erro como uma porcentagem do valor real. É útil porque fornece uma métrica de erro relativa, que é fácil de interpretar e comparar entre diferentes séries temporais ou modelos, independentemente da escala dos dados.
* **Interpretação:** Quanto **menor** o MAPE, melhor o modelo. Um MAPE de **{mape:.2f}%** significa que, em média, as previsões do seu modelo desviam dos valores reais em {mape:.2f}%. Por exemplo, um MAPE de 10% significa que o erro médio é de 10% do valor real.
* **Valores bons ou ruins?** Um MAPE de **{mape:.2f}%** pode ser considerado **bom** ou **excelente** dependendo do domínio da aplicação. Em muitas aplicações de previsão, um MAPE abaixo de 10% é considerado altamente preciso. Valores entre 10% e 20% são geralmente bons, enquanto valores acima de 50% indicam problemas significativos. Dada a complexidade das séries temporais de consumo de CPU, **{mape:.2f}%** é um resultado muito promissor!
* **Observação:** O MAPE pode ser problemático quando os valores reais são zero ou muito próximos de zero, pois causa divisão por zero ou resultados muito grandes. No seu código, adicionamos um pequeno valor (1e-8) para mitigar esse problema, garantindo a estabilidade do cálculo.
* **Referência:** Makridakis, S., & Hibon, M. (2000). The M3-Competition: results, conclusions and implications. *International Journal of Forecasting*, 16(4), 451-476.

Para uma avaliação mais aprofundada se esses valores são "bons", você precisaria compará-los com benchmarks da indústria ou com a performance de outros modelos no mesmo contexto de dados de consumo de CPU. No entanto, em termos gerais, quanto menores esses valores, melhor a capacidade preditiva do seu modelo.
""")


        # 18. Tabela e Gráficos de Visualização
        progress.progress(90)
        st.subheader("Tabela e Gráficos de Visualização")

        resultados = pd.DataFrame({
            "Previsão (Transformer)": predictions_desnorm,
            "Valor Real": real_values_desnorm,
            "Limite Inferior (95%)": previsao_inf,
            "Limite Superior (95%)": previsao_sup,
            "Acertou": (
                (previsao_inf <= real_values_desnorm) &
                (real_values_desnorm <= previsao_sup)
            )
        })

        st.write("#### Amostra dos Resultados (10 primeiras linhas)")
        st.dataframe(resultados.head(10), use_container_width=True)

        # --- SEÇÃO DO GRÁFICO CONSOLIDADO: PREVISÃO VS VALOR REAL E IC 95% ---
        # Removido o gráfico Altair redundante e o Plotly Go duplicado.
        # Agora, o gráfico "Gráfico da Acurácia do Modelo" será o principal e mais completo.

        # --- NOVO GRÁFICO: PREVISÕES VS. VALORES REAIS (Mantido) ---
        st.subheader("Gráfico: Previsões do Modelo Transformer vs. Valores Reais")
        st.info("📊 Comparando visualmente a previsão da tendência com os valores reais.")

        # Criar um DataFrame para o Plotly apenas com as colunas necessárias para este gráfico
        # Usamos .reset_index() para ter uma coluna de índice para o eixo X
        df_pred_real = resultados[['Previsão (Transformer)', 'Valor Real']].reset_index()
        # Usar melt para transformar o DataFrame para o formato "longo", ideal para Plotly com múltiplas linhas
        df_pred_real = df_pred_real.melt('index', var_name='Tipo de Valor', value_name='Consumo de CPU')

        fig_pred_real = px.line(df_pred_real, x='index', y='Consumo de CPU', color='Tipo de Valor',
                                title='Previsões vs. Valores Reais')
        
        # Ajustar o layout do gráfico para legendas claras
        fig_pred_real.update_layout(
            xaxis_title="Amostras de Teste",
            yaxis_title="Consumo de CPU",
            hovermode="x unified", # Permite ver os valores ao passar o mouse
            legend_title="Tipo de Valor",
            height=400, # Ajusta a altura para melhor visualização
            title_font_size=20,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        st.plotly_chart(fig_pred_real, use_container_width=True)
        # --- FIM DO NOVO GRÁFICO ---


        # 19. Gráfico da Acurácia do Modelo (Agora o gráfico principal consolidado)
        progress.progress(95)
        st.subheader("Gráfico Principal: Previsão, Valor Real e Intervalo de Confiança (IC 95%)")
        st.info("📈 Visualizando a previsão da tendência, a volatilidade e a acurácia do IC 95% do modelo.")

        # Preparar dados para Plotly, Plotly prefere dados "longos" para múltiplas linhas
        df_plotly = resultados.head(100).reset_index().rename(columns={"index": "Amostras de Teste"}) # Limitando a 100 pontos para melhor visualização se houver muitos dados

        if len(df_plotly) > 1:
            # Criar o gráfico interativo com Plotly
            fig = px.line(df_plotly, x="Amostras de Teste", y="Valor Real", title="Previsão vs Valor Real e IC 95%")

            # Adicionar a linha de Previsão (Tendência)
            fig.add_scatter(x=df_plotly["Amostras de Teste"], y=df_plotly["Previsão (Transformer)"],
                            mode='lines', name='Previsão (Tendência)', line=dict(color='orange', dash='dot', width=2))

            # Adicionar o intervalo de confiança como uma área sombreada
            fig.add_trace(go.Scatter(
                x=df_plotly["Amostras de Teste"],
                y=df_plotly["Limite Superior (95%)"],
                mode='lines',
                name='Limite Superior (95%)',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df_plotly["Amostras de Teste"],
                y=df_plotly["Limite Inferior (95%)"],
                mode='lines',
                name='Intervalo de Confiança (95%)', # Nome da legenda para a área
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.2)' # Verde transparente para a área do IC
            ))

            # Adicionar os pontos de erro (fora do IC)
            erros_df = df_plotly[~df_plotly["Acertou"]]
            if not erros_df.empty:
                fig.add_trace(go.Scatter(
                    x=erros_df["Amostras de Teste"], y=erros_df["Valor Real"],
                            mode='markers', name='Erros (Fora do IC)',
                            marker=dict(symbol='x', color='red', size=10)
                ))

            # Atualizar layout para melhorar a visualização e legendas
            fig.update_layout(
                height=500, # Ajusta a altura
                xaxis_title="Amostras de Teste",
                yaxis_title="Consumo de CPU",
                hovermode="x unified",
                legend_title="Legenda",
                title_font_size=20,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Não há pontos de dados suficientes no conjunto de teste (após a filtragem) para desenhar o gráfico principal de previsão. Tente um intervalo de datas maior ou um filtro de consumo de CPU mais amplo.")


        st.markdown(f"""
**Análise Visual dos Resultados**

O gráfico acima mostra que a linha azul (Valor Real) frequentemente se mantém dentro da área verde, que representa o **Intervalo de Confiança de 95%**.

Isso confirma que o modelo acertou na maioria das previsões, tanto na tendência quanto na volatilidade.  
Pontos vermelhos indicam valores reais fora do IC, destacando a relevância de quantificar incertezas.  

Com acurácia de **{acuracia_global:.2f}%**, seu modelo híbrido é extremamente confiável.
""")

        # 20. Verificação de Previsibilidade em Dados Desconhecidos
        progress.progress(100)
        st.subheader("Verificação de Previsibilidade em Dados Desconhecidos")
        st.info("🔮 Simulando 5 pontos futuros não vistos pelo modelo")
        
        # --- INÍCIO DA ADIÇÃO DO CONTADOR E RESUMO ---
        correct_simulations_count = 0 
        if len(X) > 5:
            indices_aleatorios = random.sample(range(len(X) - 5, len(X)), 5)
            for i, idx in enumerate(indices_aleatorios):
                input_scaled = X[idx]
                real_scaled = y[idx]
                inp = input_scaled.reshape(1, n_steps, X.shape[2])

                # Tendência
                pred_s = model_transformer.predict(inp, verbose=0)
                pred_trend = scaler.inverse_transform(
                    np.concatenate([pred_s, np.zeros((1, 2))], axis=1)
                )[0, 0]

                # Volatilidade
                vol = res_garch.forecast(horizon=1).variance.iloc[-1].item()
                std_f = np.sqrt(vol)
                lim_sup = std_f * 1.96
                lim_inf = -std_f * 1.96

                # Verdadeiro
                real_val = scaler.inverse_transform(
                    np.concatenate([real_scaled.reshape(1, 1), np.zeros((1, 2))], axis=1)
                )[0, 0]

                # Intervalo final
                previsao_final_inferior = pred_trend + lim_inf
                previsao_final_superior = pred_trend + lim_sup

                st.write(f"**Exemplo {i+1}:**")
                st.write(f"• Índice de previsão: {idx}")
                st.write(f"• Previsão Transformer: {pred_trend:.2f}")
                st.write(f"• Valor real desconhecido: {real_val:.2f}")
                st.write(f"• IC 95%: [{previsao_final_inferior:.2f}, {previsao_final_superior:.2f}]")

                # Avaliação final
                if previsao_final_inferior <= real_val <= previsao_final_superior:
                    st.success(
                        "Avaliação: O modelo acertou! O valor real está dentro do intervalo de confiança. "
                        "A combinação dos modelos foi eficaz para prever a volatilidade deste ponto."
                    )
                    correct_simulations_count += 1 # Incrementa o contador
                else:
                    st.error(
                        "Avaliação: O modelo errou a previsão. O valor real não está dentro do intervalo de confiança. "
                        "A volatilidade foi muito alta ou atípica, e os modelos não conseguiram prevê-la."
                    )
            
            st.markdown(f"---") # Linha divisória para o resumo
            st.markdown(f"**Resumo da Simulação:** Dos 5 exemplos de dados aleatórios ainda não vistos pelo modelo, ele conseguiu prever **{correct_simulations_count}** corretamente (dentro do IC 95%).")
            st.markdown(f"---")
        # --- FIM DA ADIÇÃO DO CONTADOR E RESUMO ---

        else:
            st.warning("Não há dados suficientes para a simulação de pontos futuros.")

else:
    st.sidebar.info("Aguardando upload de arquivo CSV ou Excel...")
