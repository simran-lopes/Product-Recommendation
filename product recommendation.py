
"""
Created on Mon Mar 22 12:49:41 2021

@author: simran lopes
"""
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# reading the dataset
dataset = pd.read_csv('OnlineRetail.csv')
dataset.head(6)

# PRODUCT RECOMMENDATION section 2
# FINDIG MOST POPULAR PRODUCT BASED ON QUANTITY PART 1
popular_products = pd.DataFrame(dataset.groupby('Description')['Quantity'].count())
# popular_products = dataset.groupby('Description').agg({'Quantity': lambda x: x.sum()})
most_popular = popular_products.sort_values('Quantity', ascending=False)
# Top 10 PRODUCTS
most_popular.head(10)
most_popular.head(30).plot(kind = "bar")


# PART 2 BASED ON PRODCT HISTORY
#SUBSET OFTHE DATASET
dataset1 = dataset.head(10000)


# making the UTILITY MATRIX 
quantity_utility_matrix = dataset1.pivot_table(values='Quantity', index='CustomerID',
                                               columns='StockCode', fill_value=0)
quantity_utility_matrix.head()


# the utility matrix obtaned above is sparce, hence we filled up the unknown values wth 0.
quantity_utility_matrix.shape
#  transporing the matrix
X = quantity_utility_matrix.T
X.head()
X.shape

# Decomposing the Matrix
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape
# Correlation Matrix
correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape
# Isolating Product ID "20760" from the Correlation Matrix
# Assuming the customer buys Product ID "20760" (randomly chosen)
X.index[99]
i = "20760"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID

# Correlation for all items with the item purchased by this customer
 # based on items rated by other customers people who bought the same product
correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape
# Recommending top 10 highly correlated products in sequence
Recommend = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer
Recommend.remove(i) 
# SHOWING TOP 10 ITEMS
Recommend[0:9]

# Recommendation System part 3
# data reading n cleaning
Description = pd.read_csv('OnlineRetail.csv')
Description.shape
Description = Description.dropna()
Description = Description.query("Country=='United Kingdom'").reset_index(drop=True)
Description.shape
Description.head()

product_descriptions1 = Description.head(500)
product_descriptions1.iloc[:,1]

product_descriptions1["Description"].head(10)
# Feature extraction from product description

# Converting the text in product description into numerical data for analysis
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions1["Description"])
X1

X=X1
# Visualizing product clusters in subset of data
kmeans = KMeans(n_clusters = 10, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)
plt.plot(y_kmeans, ".")
plt.show()
#  making clusters
def print_cluster(i):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
# Top words in each cluster based on product description   
true_k = 10

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print_cluster(i)
# Predicting clusters based on key search words
def show_recommendations(product):
    #print("Cluster ID:")
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    #print(prediction)
    print_cluster(prediction[0])
# showing the product belonging to their recommended cluster

show_recommendations("KNITTED UNION FLAG HOT WATER BOTTLE")

    






































