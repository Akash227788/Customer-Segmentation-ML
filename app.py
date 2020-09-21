import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import spacy

import sklearn.cluster as cluster

sns.set_style()
plt.style.use('fivethirtyeight')


import streamlit as st

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering



#@st.cache
#URL = "A:\\Fde_final_app\\Mall_Customers.csv"
customer = pd.read_csv('Mall_Customers.csv')
#st.write(customer.head())

customer.rename(columns={'Annual Income (k$)' : 'Income', 'Spending Score (1-100)' : 'Spending_Score'}, inplace = True)

st.title("Customer segmentation")

#st.radio("Navigation", ["Home", "About Us"])
st.sidebar.title("Our Model")

st.sidebar.info("""
	This is a machine learning project, 
	Where we analyze and cluster a customers data
	based on their spending habits and their income.
	""")

st.markdown("""
	## About Data
	this file contains the basic information (ID, age, gender, income, spending score) about the customers
	""")

st.markdown("""
	## Problem Statement
	You own the mall and want to understand the customers like who can be easily converge [Target Customers] so that the sense can be given to marketing team and plan the strategy accordingly.
	""")


st.markdown("""
	## Target
	By the end of this case study , you would be able to answer below questions.
	1- How to achieve customer segmentation using machine learning algorithm (KMeans Clustering) in Python in simplest way.
	2- Who are your target customers with whom you can start marketing strategy [easy to converse]
	3- How the marketing strategy works in real world
	""")


st.cache(persist = True)
st.sidebar.markdown("""
	## Customer Data
	""")

if st.sidebar.checkbox("RAW DATA"):
	st.write(customer)

# This shows male and female ratio


st.sidebar.markdown("""
	## Data Analysis
	""")

st.markdown("""
	## Data Analysis
	 Data is one of the important features of every organization because it helps business leaders to make decisions based on facts, statistical numbers and trends. Due to this growing scope of data, data science came into picture which is a multidisciplinary field. It uses scientific approaches, procedure, algorithms, and framework to extract the knowledge and insight from a huge amount of data.
	""")

if st.sidebar.checkbox("Data Analysis"):
	st.subheader("Pie chart")
	labels = ['Female', 'Male']
	size = customer['Gender'].value_counts()
	colors = ['lightgreen', 'orange']
	explode = [0, 0.1]

	plt.rcParams['figure.figsize'] = (5, 5)
	plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
	plt.title('Gender', fontsize = 20)
	plt.axis('off')
	plt.legend()
	st.pyplot()

	st.subheader("Second chart")
	plt.figure(figsize = (18,8))
	ax = sns.countplot(x="Age", hue="Gender", data=customer)
	st.pyplot()

	st.subheader("Third chart")
	plt.figure(figsize = (18,8))
	ax = sns.countplot(x="Income", hue="Gender", data=customer)
	st.pyplot()


	plt.figure(figsize = (15,20))
	ax = sns.countplot(y="Spending_Score", hue="Gender", data=customer)	
	st.pyplot()

	sns.pairplot(customer, vars= ['Age', 'Income', 'Spending_Score'], hue="Gender")
	fig = plt.gcf()
	fig.set_size_inches(17,10)	
	st.pyplot()

	
# Cluster for income V/S Annual Income		
st.markdown(""" 
	## *K Means Clustering Model*

	K-means clustering algorithm computes the centroids and iterates until we it finds optimal centroid. It assumes that the number of clusters are already known. It is also called flat clustering algorithm. The number of clusters identified from data by algorithm is represented by ‘K’ in K-means.

	""")
st.sidebar.markdown("""
	## K Means Algorithm for Income V/S Spending Score
	""")

if st.sidebar.checkbox("K Means Algorithm"):
	customer_short = customer[['Spending_Score','Income']]
	K=range(1,12)
	wss = []
	for k in K:
	    kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
	    kmeans=kmeans.fit(customer_short)
	    wss_iter = kmeans.inertia_
	    wss.append(wss_iter)
	mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})
	

	sns.lineplot(x = 'Clusters', y = 'WSS', data = mycenters, marker="+")
	plt.title("Spending score x Income")

	fig = plt.gcf()
	fig.set_size_inches(15,5)
	st.pyplot()
	# We get 5 Clusters

	# Model for spending score V/S Income
	kmeans = cluster.KMeans(n_clusters=5 ,init="k-means++")
	kmeans = kmeans.fit(customer[['Spending_Score','Income']])	

	customer['Clusters'] = kmeans.labels_

	customer.to_csv('mallClusters.csv', index = False)

	sns.pairplot(customer, vars= ['Age', 'Income', 'Spending_Score'], hue="Clusters", height=4)
	st.pyplot()

	y = customer.iloc[:, [2, 4]].values

	from sklearn.cluster import KMeans

	wcss = []
	for i in range(1, 11):
	    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
	    kmeans.fit(y)
	    wcss.append(kmeans.inertia_)	
	kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
	ymeans = kmeans.fit_predict(y)

	plt.rcParams['figure.figsize'] = (12, 10)
	plt.title('Cluster of Annual Income', fontsize = 30)

	plt.scatter(y[ymeans == 0, 0], y[ymeans == 0, 1], s = 100, c = 'pink', label = 'Priority Customers' )
	plt.scatter(y[ymeans == 1, 0], y[ymeans == 1, 1], s = 100, c = 'orange', label = 'Target Customers(High income)')
	plt.scatter(y[ymeans == 2, 0], y[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Usual Customers')
	plt.scatter(y[ymeans == 3, 0], y[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Less Income)')
	plt.scatter(y[ymeans == 4, 0], y[ymeans == 4, 1], s = 100, c = 'blue', label = 'Moderately spending customers')
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'Black', marker ='*')

	plt.style.use('fivethirtyeight')
	plt.xlabel('Annual Income')
	plt.ylabel('Spending Score (1-100)')
	plt.legend()
	plt.grid()
	st.pyplot()    

st.cache()
# Spending score and age clustering

st.sidebar.markdown("""
	## K Means Algorithm for age V/S spending score
	""")

if st.sidebar.checkbox("K Means Algorithm ") :
	customer_range = customer[['Spending_Score','Age']]

	K=range(1,12)
	wss = []
	for k in K:
	    kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
	    kmeans=kmeans.fit(customer_range)
	    wss_iter = kmeans.inertia_
	    wss.append(wss_iter)

	mycenters1 = pd.DataFrame({'Clusters1' : K, 'WSS' : wss})
	
	sns.lineplot(x = 'Clusters1', y = 'WSS', data = mycenters1, marker="+")
	plt.title("Spending score x Age")

	fig = plt.gcf()
	fig.set_size_inches(15,5)
	# We get 4 Clusters    
	st.pyplot()

	x = customer.iloc[:, [2, 4]].values
	from sklearn.cluster import KMeans

	wcss = []
	for i in range(1, 11):
	    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
	    kmeans.fit(x)
	    wcss.append(kmeans.inertia_)	
	kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
	ymeans = kmeans.fit_predict(x)

	plt.rcParams['figure.figsize'] = (10, 10)
	plt.title('Cluster of Ages', fontsize = 30)

	plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'Usual Customers' )
	plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')
	plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')
	plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black')

	plt.style.use('fivethirtyeight')
	plt.xlabel('Age')
	plt.ylabel('Spending Score (1-100)')
	plt.legend()
	plt.grid()
	st.pyplot()  



st.sidebar.markdown("""
	## Hierarchical Clustering
	""")

if st.sidebar.checkbox("Hierarchical Clustering Algorithm") :
	X = customer.iloc[:,[3,4]].values
	plt.figure(figsize=(15,6))
	plt.title('Dendrogram')
	plt.xlabel('Customers')
	plt.ylabel('Euclidean distances')
	plt.hlines(y=190,xmin=0,xmax=2000,lw=3,linestyles='--')
	plt.text(x=900,y=220,s='Horizontal line crossing 5 vertical lines',fontsize=20)
	#plt.grid(True)
	dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
	st.pyplot()

	X = customer.iloc[:,[3,4]].values
	hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
	y_hc = hc.fit_predict(X)
	plt.figure(figsize=(12,7))
	plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful-Cluster1')
	plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard-Cluster2')
	plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target group-Cluster3')
	plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'orange', label = 'Careless-Cluster4')
	plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible-Cluster5')
	plt.title('Clustering of customers',fontsize=20)
	plt.xlabel('Annual Income (k$)',fontsize=16)
	plt.ylabel('Spending Score (1-100)',fontsize=16)
	plt.legend(fontsize=16)
	plt.grid(True)
	plt.axhspan(ymin=60,ymax=100,xmin=0.4,xmax=0.96,alpha=0.3,color='yellow')
	st.pyplot()



