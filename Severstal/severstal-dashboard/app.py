import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import faiss
import os
import math
import warnings

# --- Настройки окружения ---
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ограничиваем выполнение только на CPU

st.set_page_config(page_title="FAISS Иерархическая кластеризация", layout="wide")

# --- Заголовок приложения ---
st.title("🔬 Интерактивная иерархическая кластеризация (FAISS + Decision Tree)")

# --- Загрузка файла пользователем ---
uploaded_file = st.file_uploader("Загрузите Excel или CSV файл", type=["xlsx", "csv"])

if uploaded_file:
    # --- Определяем тип файла ---
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"✅ Файл загружен: {uploaded_file.name}")
    st.write(f"Исходная форма данных: {df.shape}")

    # --- Панель настроек ---
    with st.sidebar:
        st.header("⚙️ Настройки анализа")

        # --- Опция выбора строк ---
        use_all_rows = st.checkbox("📋 Рассмотреть все строки", value=True)

        if not use_all_rows:
            max_rows = st.number_input(
                "Максимум строк для анализа",
                min_value=10,
                max_value=len(df),
                value=min(1000, len(df)),
                step=10
            )
            df = df.head(max_rows)
        else:
            max_rows = len(df)

        # --- Опция выбора столбцов ---
        use_all_columns = st.checkbox("🧩 Рассмотреть все столбцы", value=True)

        if not use_all_columns:
            max_features = st.number_input(
                "Максимум столбцов для анализа",
                min_value=1,
                max_value=len(df.columns),
                value=min(20, len(df.columns)),
                step=1
            )

            selected_cols = st.multiselect(
                "Выберите столбцы для анализа",
                df.columns.tolist(),
                default=df.columns[:max_features].tolist()
            )
        else:
            selected_cols = df.columns.tolist()

        # --- Выбор таргета ---
        target_col = st.selectbox(
            "🎯 Выберите таргет (опционально)",
            options=["<нет>"] + df.columns.tolist(),
            index=0
        )
        if target_col == "<нет>":
            target_col = None

        # --- Флаги включения EDA и корреляции ---
        show_eda = st.checkbox("Показать EDA (исследовательский анализ данных)", value=True)
        show_corr = st.checkbox("Показать корреляционную матрицу", value=True)

        # --- Флаг удаления выбросов ---
        remove_outliers = st.checkbox("Удалить выбросы", value=False)

    # --- Очистка данных ---
    df = df[selected_cols].dropna()
    st.write(f"После удаления пропусков: {df.shape}")

    # --- Определение типов данных ---
    df_num_col = df.select_dtypes(include=[np.number]).columns.tolist()
    df_cat_col = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Удаление выбросов (если выбрано) ---
    if remove_outliers and df_num_col:
        st.warning("⚠️ Активировано удаление выбросов.")
        before = df.shape[0]
        for col in df_num_col:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = df.shape[0]
        st.write(f"Удалено выбросов: {before - after} строк ({round((before - after) / before * 100, 2)}%)")
        st.write(f"После удаления выбросов: {df.shape}")

    if target_col:
        st.success(f"🎯 Выбран таргет: {target_col}")
    else:
        st.info("Кластеризация выполняется без таргета.")

    # --- EDA (если включен) ---
    if show_eda:
        st.header("📊 Исследовательский анализ данных")

        # Визуализация числовых признаков
        if df_num_col:
            n_cols = 2
            n_rows = len(df_num_col)
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 4 * n_rows))
            axes = axes.reshape(-1, 2)
            for i, col in enumerate(df_num_col):
                sns.histplot(df[col], bins=30, kde=True, ax=axes[i][0], color='skyblue')
                axes[i][0].set_title(f'Гистограмма: {col}')
                sns.boxplot(x=df[col], ax=axes[i][1], color='lightcoral')
                axes[i][1].set_title(f'Boxplot: {col}')
            plt.tight_layout()
            st.pyplot(fig)

        # Визуализация категориальных признаков
        if df_cat_col:
            top_n = 5
            n_cols = 3
            n_rows = math.ceil(len(df_cat_col) / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            axes = axes.flatten()
            for i, col in enumerate(df_cat_col):
                top_cats = df[col].value_counts().nlargest(top_n).index
                df_plot = df.copy()
                df_plot[col] = df_plot[col].apply(lambda x: x if x in top_cats else 'Other')
                count_data = df_plot[col].value_counts()
                percent_data = df_plot[col].value_counts(normalize=True) * 100
                sns.barplot(x=count_data.index, y=count_data.values, ax=axes[i],
                            palette=sns.color_palette("coolwarm", n_colors=len(count_data)))
                axes[i].set_title(f'Top {top_n} категорий: {col}')
                axes[i].tick_params(axis='x', rotation=45)
                for j, val in enumerate(count_data.values):
                    axes[i].text(j, val + val * 0.01, f'{percent_data.values[j]:.1f}%', ha='center')
            for j in range(len(df_cat_col), len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            st.pyplot(fig)

    # --- Корреляционный анализ (если включен) ---
    if show_corr and df_num_col:
        st.subheader("📈 Корреляционная матрица")
        corr = df[df_num_col].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # --- Кодирование категориальных признаков ---
    cat_encoders = []
    for col in df_cat_col:
        if df[col].nunique() < 20:
            cat_encoders.append((col, OneHotEncoder(handle_unknown='ignore'), [col]))
        else:
            cat_encoders.append((col, OrdinalEncoder(), [col]))

    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), df_num_col)] + cat_encoders,
        remainder='drop'
    )

    X = preprocessor.fit_transform(df)
    X = X.toarray() if hasattr(X, 'toarray') else X

    # --- Автоматический выбор числа кластеров ---
    st.header("🤖 Кластеризация")

    def optimal_kmeans_clusters(X, max_k=10):
        best_k, best_score = 2, -1
        for k in range(2, min(max_k, X.shape[0]-1)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    optimal_k = optimal_kmeans_clusters(X)
    st.success(f"Оптимальное число кластеров: {optimal_k}")

    # --- FAISS кластеризация ---
    X = np.ascontiguousarray(X.astype('float32'))
    kmeans = faiss.Kmeans(d=X.shape[1], k=optimal_k, niter=25, verbose=False, gpu=False)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    df["cluster"] = I.flatten()

    # --- Обучаем Decision Tree ---
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X, I.flatten())

    # --- PCA визуализация ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["cluster"], palette="tab10", ax=ax)
    plt.title("FAISS Кластеры (PCA проекция)")
    st.pyplot(fig)

    # --- Количество объектов по кластерам ---
    st.subheader("📦 Количество объектов в кластерах")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x="cluster", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

    # --- Распределение таргета по кластерам ---
    if target_col:
        st.subheader("🎯 Распределение таргета по кластерам")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="cluster", y=target_col, data=df, palette="Set3", ax=ax)
        st.pyplot(fig)

    # --- Средние значения признаков по кластерам ---
    st.subheader("📊 Средние значения признаков по кластерам")
    cluster_means = df.groupby("cluster")[df_num_col].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    # --- Дерево решений ---
    st.subheader("🌳 Дерево решений (интерпретация кластеров)")
    feature_names = preprocessor.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        tree,
        filled=True,
        feature_names=feature_names,
        class_names=[f"Cluster {i}" for i in np.unique(I.flatten())],
        rounded=True,
        fontsize=8,
        ax=ax
    )
    st.pyplot(fig)

else:
    st.info("👆 Загрузите файл для начала анализа.")