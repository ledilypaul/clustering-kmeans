import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import statistics as stat
from PIL import Image
import random as rnd

from numpy.core.fromnumeric import argmin, mean


image_path = "Images"

perroquet = Image.open(image_path+"/perroquet_couleur.png").convert('RGB')
perroquetg = perroquet.convert('L')

canyon = Image.open(image_path+"/canyon.jpg")




# Algorithme des K means
def kmeans(X, k):
    centroids =[]
    # Initialisation aléatoire des centroide
    while(len(centroids)<k):
        r1 = rnd.randint(0,255)
        r2 = rnd.randint(0,255)
        r3 = rnd.randint(0,255)
        R = np.array([r1,r2,r3])
        ids = map(id,centroids)
        # Centroids n'a pas déjà été généré alors on l'ajoute
        if id(R) not in ids:
            centroids.append(R)	
    while True:
        new_centroids = [np.array([0,0,0]) for x in centroids]
        new_centroids_count = [0 for x in centroids]

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                #On regarde pour chaque pixel de l'image quel est le centroid le plus proche
                index = argmin([np.linalg.norm(X[i][j]-c) for c in centroids])

                # On ajoute 1 au nombre d'éléments dans le cluster
                new_centroids[index] += X[i][j]
                new_centroids_count[index] += 1

        #On calcule la moyenne entre le nombre de  points dans le cluster et la somme des valeurs du cluster au nouveau barycentre 
        new_centroids = [m/(c+1) for m,c in zip(new_centroids,new_centroids_count)]

        #Condition d'arret
        if max([np.linalg.norm(c-n) for c,n in zip(centroids,new_centroids)])<1:
            break
        centroids = new_centroids
    return new_centroids


#Fonction de classification des K means
def classifyByKmeans(X, k):
    X = np.array(X)
    # Renvoie la liste des centroids/barycentres
    centroids = kmeans(X, k)
    print(centroids)
    print(len(centroids))
#   Pour chaque pixel, on va vérfier quel est le centroids le plus proche. Et nous allons remplacer la valeur RGB du pixel par celle du centroid 
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = centroids[argmin([np.linalg.norm(X[i][j]-c) for c in centroids])]
    return X


# Fonction d'affichage du graphique 3D
def show_image_pixel(img):
    data = np.array(img)
    dict_color = {}

    def _hash(c):
        return str(hash(str(c)))

    data_flatten = data.reshape((data.shape[0] * data.shape[1], 3))

    for c in data_flatten:
        h = _hash(c)
        dict_color[h] = c
    
    colors = np.array([x[1] for x in dict_color.items()])

    x, y, z = [], [], []

    for c in colors:
        x.append(c[0])
        y.append(c[1])
        z.append(c[2])
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel("RED")
    ax.set_ylabel("GREEN")
    ax.set_zlabel("BLUE")
    ax.scatter(x, y, z, c=colors/255.0)

    plt.show()

################################################################################################################################################
# Affichage de l'image pour K = 4

data = classifyByKmeans(perroquet,4)
plt.imshow(data)
plt.title("Image avec K = 4")
plt.show()

# Affiche la représentation 3D en lien avec l'image
show_image_pixel(data)

################################################################################################################################################
# Affichage de l'image pour K = 8

# data = classifyByKmeans(perroquet,8)
# plt.imshow(data)
# plt.title("Image avec K = 8")
# plt.show()

# Affiche la représentation 3D en lien avec l'image
# show_image_pixel(data)

################################################################################################################################################
# Affichage de l'image pour K = 16

# data = classifyByKmeans(perroquet,16)
# plt.imshow(data)
# plt.title("Image avec K = 16")
# plt.show()

# Affiche la représentation 3D en lien avec l'image
# show_image_pixel(data)

################################################################################################################################################
# Affichage de l'image pour K = 32

# data = classifyByKmeans(perroquet,32)
# plt.imshow(data)
# plt.title("Image avec K = 32")
# plt.show()

# Affiche la représentation 3D en lien avec l'image
# show_image_pixel(data)