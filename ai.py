# K-MEANS CLUSTERING TOPIK ENTERTAINMENT
# Contoh: Mengelompokkan film berdasarkan rating, penonton, dan budget

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --------------------------
# 1. MEMBUAT DATASET FILM
# --------------------------
data = {
    'Judul_Film': [
        'Sharelock', 'Maria', 'Joker', 'Minions', 'Avatar',
        'Spiderman', 'Toy Story', 'Batman', 'Inception', 'Moana'
    ],
    'Rating': [8.4, 7.4, 8.5, 6.9, 7.8, 7.5, 8.0, 9.0, 8.7, 7.6],
    'Jumlah_Penonton_Juta': [140, 120, 90, 110, 150, 130, 95, 85, 100, 115],
    'Budget_JutaUSD': [220, 150, 60, 74, 237, 200, 120, 185, 160, 145]
}

df = pd.DataFrame(data)

# --------------------------
# 2. MENENTUKAN FITUR
# --------------------------
X = df[['Rating', 'Jumlah_Penonton_Juta', 'Budget_JutaUSD']]

# --------------------------
# 3. MENJALANKAN K-MEANS
# --------------------------
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

print(df)

# --------------------------
# 4. VISUALISASI (2D)
# --------------------------
plt.figure(figsize=(8,5))
plt.scatter(df['Rating'], df['Jumlah_Penonton_Juta'], c=df['Cluster'])
plt.xlabel("Rating")
plt.ylabel("Jumlah Penonton (juta)")
plt.title("Clustering Film (Entertainment) dengan K-Means")
plt.show()


