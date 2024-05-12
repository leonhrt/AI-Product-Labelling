__authors__ = ['1679933','1689435']
__group__ = '100'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import KNN
import Kmeans
import numpy as np


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    def retrieval_by_color(images, labels, query_color):
        # si query_color no és una llista (és un sol string ja que el query és d'un sol color), es converteix a llista
        # per poder-la iterar
        if type(query_color) != list:
            query_color = [query_color]

        # crear llista que contindrà les imatges que coincideixin en colors
        retrieved_images = []
        # iterar per les etiquetes de colors i imatges per trobar coincidència amb query_color
        for img, label in zip(images, labels):
            # verificar si els colors de l'etiqueta coincideixen amb alguna query_color, np.char.lower() converteix a minúscules
            # per no tenir errors en la comparació, any() retorna True si algun dels elements de query_color és igual
            # que l'etiqueta
            if any(query in np.char.lower(label) for query in np.char.lower(query_color)):
                # si es troba imatge que coincideix en colors amb query_color, s'afegeix a retrieved_images[]
                retrieved_images.append(img)
        # es retorna retrieves_images[], que ja conté totes les imatges coincidents en color amb query_color
        return retrieved_images


    def retrieval_by_shape(images, labels, query_shape):
        # crear llista que contindrà les imatges que coincideixin en colors
        retrieved_images = []
        # iterar per les etiquetes de forma i imatges per trobar coincidència amb query_shape
        for img, label in zip(images, labels):
            # verificar si la forma de l'etiqueta coincideix amb query_shape, .lower() converteix a minúscules
            # per no tenir errors en la comparació
            if query_shape.lower() == label.lower():
                # si es troba imatge que coincideix en forma amb query_shape, s'afegeix a retrieved_images[]
                retrieved_images.append(img)
        # es retorna retrieves_images[], que ja conté totes les imatges coincidents en forma amb query_shape
        return retrieved_images


    def retrieval_combined(images, color_labels, shape_labels, query_color, query_shape):
        # si query_color no és una llista (és un sol string ja que el query és d'un sol color), es converteix a llista
        # per poder-la iterar
        if type(query_color) != list:
            query_color = [query_color]

        # crear llista que contindrà les imatges que coincideixin
        retrieved_images = []
        # iterar per les llistes d'imatges, etiquetes de colors i etiquetes de formes a la vegada
        for img, color_label, shape_label in zip(images, color_labels, shape_labels):
            # comprovar si l'etiqueta de forma d'aquella imatge coincideix amb la query_shape
            if query_shape.lower() == shape_label.lower():
                # si coincideix en forma, comprovar si alguna de les query_color és l'etiqueta de color
                if any(query in np.char.lower(color_label) for query in np.char.lower(query_color)):
                    # si coincideix tant en forma com en color amb query_shape i query_color, afegir a
                    # retrieved_images[]
                    retrieved_images.append(img)
        # es retorna retrieves_images[], que ja conté totes les imatges coincidents en forma i color
        return retrieved_images


    # ------------- set up ---------------

    knn = KNN.KNN(train_imgs, train_class_labels)

    shape_labels = knn.predict(test_imgs, 5)

    imgs = test_imgs
    color_labels = []
    options = {}
    for img in imgs:
        km = Kmeans.KMeans(img, 1, options)
        km.fit()
        colors = Kmeans.get_colors(km.centroids)
        color_labels.append(colors)

    # ------------- qualitative analysis ---------------

    color = retrieval_by_color(test_imgs, color_labels, 'white')
    visualize_retrieval(color, 20)

    shape = retrieval_by_shape(test_imgs, shape_labels, 'Flip FLOPs')
    visualize_retrieval(shape, 20)

    combined = retrieval_combined(test_imgs, color_labels, shape_labels, 'PInk', 'HandBAGs')
    visualize_retrieval(combined, 20)

