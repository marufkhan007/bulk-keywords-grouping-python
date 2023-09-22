import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Define a main function to encapsulate the main functionality
def main():
    # Load the Excel data
    df = pd.read_excel(r'C:\Users\OMY PC\Downloads\Documents\comeon')  # Provide the correct file path

    # Define the range of values for num_clusters
    min_clusters = 5
    max_clusters = 20

    # Preprocess keywords
    stopwords_set = set(stopwords.words('english'))
    df['Cleaned_Keyword'] = df['Keyword'].apply(
        lambda x: ' '.join([word for word in re.sub(r'[^a-zA-Z\s]', '', x).split() if word not in stopwords_set]).lower()
    )

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Keyword'])

    # Elbow Method to find an optimal number of clusters
    wcss = []
    silhouette_scores = []
    for num_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(tfidf_matrix)
        wcss.append(kmeans.inertia_)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(tfidf_matrix, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    # Plot the Elbow Method results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(min_clusters, max_clusters + 1), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')

    # Plot the Silhouette Score results
    plt.subplot(1, 2, 2)
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores)
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

    # Based on the results, choose an appropriate value for num_clusters
    chosen_num_clusters = 10  # You can adjust this based on the plot or domain knowledge

    # Perform K-Means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=chosen_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df['Cluster'] = kmeans.fit_predict(tfidf_matrix)

    # Export the results to a new Excel file
    df.to_excel('grouped_keywords_advanced.xlsx', index=False)

# Ensure the code is executed when you run the script
if __name__ == "__main__":
    main()

#
#