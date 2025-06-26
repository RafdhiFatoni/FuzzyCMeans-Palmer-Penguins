import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        filepath (str): Path ke file CSV.

    Returns:
        pd.DataFrame: DataFrame hasil pembacaan file.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat dataset: {e}")
        return pd.DataFrame()

def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris yang memiliki missing value dari DataFrame.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa missing value.
    """
    return df.dropna()

def remove_noisy_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris dengan nilai gender selain 'male' atau 'female'.
    Args:
        df (pd.DataFrame): DataFrame input.
    Returns:
        pd.DataFrame: DataFrame tanpa noisy data pada kolom gender.
    """
    return df[df['sex'].isin(['MALE', 'FEMALE'])]

def plot_column_distributions(df: pd.DataFrame) -> None:
    """
    Membuat visualisasi distribusi data untuk semua kolom dalam DataFrame.
    Args:
        df (pd.DataFrame): DataFrame input.
    """
    num_cols = len(df.columns)
    n_cols = 3
    n_rows = (num_cols + n_cols - 1) // n_cols
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    for idx, col in enumerate(df.columns, 1):
        plt.subplot(n_rows, n_cols, idx)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].hist(bins=30, edgecolor='black')
        else:
            df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribusi {col}')
        plt.xlabel(col)
        plt.ylabel('Frekuensi')
    plt.tight_layout()
    plt.show()

def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Membuat box-plot terpisah untuk setiap kolom numerik dalam DataFrame.
    Args:
        df (pd.DataFrame): DataFrame input.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    num_cols = len(numeric_cols)
    if num_cols == 0:
        print("Tidak ada kolom numerik untuk ditampilkan.")
        return
    n_rows = (num_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(6 * 2, 4 * n_rows))
    axes = axes.flatten()
    for idx, col in enumerate(numeric_cols):
        df.boxplot(column=col, ax=axes[idx])
        axes[idx].set_title(f'Box-Plot {col}')
        axes[idx].set_ylabel('Nilai')
    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Membuat heatmap korelasi untuk semua kolom numerik dalam DataFrame.
    Args:
        df (pd.DataFrame): DataFrame input.
    """
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        print("Tidak cukup kolom numerik untuk membuat heatmap korelasi.")
        return
    corr = numeric_df.corr()
    plt.figure(figsize=(1.2 * len(corr.columns), 1.2 * len(corr.columns)))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Heatmap Korelasi Kolom Numerik')
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_pie_chart(df: pd.DataFrame, column: str) -> None:
    """
    Membuat chart lingkaran (pie chart) untuk proporsi nilai pada kolom tertentu.
    Args:
        df (pd.DataFrame): DataFrame input.
        column (str): Nama kolom yang ingin divisualisasikan.
    """
    if column not in df.columns:
        print(f"Kolom '{column}' tidak ditemukan dalam DataFrame.")
        return
    counts = df[column].value_counts()
    plt.figure(figsize=(6, 6))
    counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title(f'Proporsi {column}')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()




print("Data preprocessing module loaded successfully.")

