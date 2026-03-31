# ================= IMPORTS =================
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import mode

# ================= GLOBAL VARIABLES =================
X, y = None, None
y_encoded = None
kmeans = None
mapped_labels = None

# ================= FUNCTIONS =================

def load_data():
    global X, y, y_encoded
    X, y = load_iris(return_X_y=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    messagebox.showinfo("Success", "Iris Dataset Loaded!")

def elbow_method():
    if X is None:
        messagebox.showerror("Error", "Load dataset first!")
        return

    distortions = []
    K = range(1, 10)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.figure()
    plt.plot(K, distortions, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.show()

def run_kmeans():
    global kmeans, mapped_labels

    if X is None:
        messagebox.showerror("Error", "Load dataset first!")
        return

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    # Map clusters
    labels = np.zeros_like(y_pred)

    for i in range(3):
        mask = (y_pred == i)
        labels[mask] = mode(y_encoded[mask], keepdims=True)[0]

    mapped_labels = labels

    acc = accuracy_score(y_encoded, mapped_labels)
    cm = confusion_matrix(y_encoded, mapped_labels)

    result_text.set(f"Accuracy: {acc:.2f}\n\nConfusion Matrix:\n{cm}")

def plot_clusters():
    if mapped_labels is None:
        messagebox.showerror("Error", "Run K-Means first!")
        return

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=mapped_labels)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                marker='X', s=200)

    plt.title("Clusters (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

# ================= GUI =================
root = tk.Tk()
root.title("K-Means Iris Mini Project")
root.geometry("500x400")

title = tk.Label(root, text="K-Means Clustering GUI", font=("Arial", 16))
title.pack(pady=10)

btn_load = tk.Button(root, text="Load Dataset", command=load_data, width=20)
btn_load.pack(pady=5)

btn_elbow = tk.Button(root, text="Elbow Method", command=elbow_method, width=20)
btn_elbow.pack(pady=5)

btn_run = tk.Button(root, text="Run K-Means", command=run_kmeans, width=20)
btn_run.pack(pady=5)

btn_plot = tk.Button(root, text="Show Clusters", command=plot_clusters, width=20)
btn_plot.pack(pady=5)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, justify="left")
result_label.pack(pady=20)

# ================= RUN =================
root.mainloop()
