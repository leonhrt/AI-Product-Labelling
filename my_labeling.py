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
        # crear llista que contindrà les imatges que coincideixin en colors
        retrieved_images = []
        # iterar per les etiquetes de colors per trobar coincidència amb query_color
        for i, img_labels in enumerate(labels):
            # verificar si els colors de l'etiqueta coincideixen amb query_color, lower() converteix a minúscules
            # per no tenir errors en la comparació, all() retorna True si tots els elements de cada vector coincideixen
            if any(query_color.lower() in label.lower() for label in img_labels):
                # si es troba imatge que coincideix en colors amb query_color, s'afegeix a retrieved_images[]
                retrieved_images.append(images[i])
        # es retorna retrieves_images[], que ja conté totes les imatges coincidents en color amb query_color
        return retrieved_images


    def retrieval_by_shape(images, shapes, query_shape):
        # crear llista que contindrà les imatges que coincideixin en forma
        retrieved_images = []
        # iterar per les etiquetes de formes per trobar coincidència amb query_shape
        for i, img_shape in enumerate(shapes):
            # verificar si la forma de l'etiqueta coincideix amb query_shape, lower() converteix a minúscules
            # per no tenir errors en la comparació
            if query_shape.lower() in img_shape.lower():
                # si es troba imatge que coincideix en forma amb query_shape, s'afegeix a retrieved_images[]
                retrieved_images.append(images[i])
        # es retorna retrieves_images[], que ja conté totes les imatges coincidents en forma amb query_shape
        return retrieved_images


    def retrieval_combined(images, color_labels, shape_labels, color_question, shape_question):
        # buscar les coincidències en color
        retrieved_color = np.array(retrieval_by_color(images, color_labels, color_question))
        # buscar les coincidències en forma
        retrieved_shape = np.array(retrieval_by_shape(images, shape_labels, shape_question))
        # intersecció dels arrays anteriors per obtenir un vector amb les imatges presents tant en el vector
        # de coincidents per forma com en el coincidents per color
        retrieved_combined = list(np.intersect1d(retrieved_color, retrieved_shape))
        return retrieved_combined

        #retrieved_images = []
        #for i, (img, color_label, shape_label) in enumerate(zip(images, color_labels, shape_labels)):
        #    if shape_question.lower() in shape_label.lower():
        #        for color in color_label:
        #            if color_question.lower() == color.lower():
        #                retrieved_images.append(img)
        #                break
        #return retrieved_images


    # ------------- set up ---------------

    # train the KNN algorithm
    knn = KNN.KNN(train_imgs, train_class_labels)

    # predict test_class_labels
    predicted_class_labels = knn.predict(test_imgs, 5)  # why is K 5?

    imgs = test_imgs
    tags = []
    options = {}
    for img in imgs:
        km = Kmeans.KMeans(img, 1, options)
        km.fit()
        colors = Kmeans.get_colors(km.centroids)
        tags.append(colors)

    # ------------- qualitative analysis ---------------

    pink = retrieval_by_color(test_imgs, tags, 'Pink')
    visualize_retrieval(pink, 10)

    jeans = retrieval_by_shape(test_imgs, predicted_class_labels, 'Jeans')
    visualize_retrieval(jeans, 10)

    pink_jeans = retrieval_combined(test_imgs, tags, predicted_class_labels, 'Pink', 'Jeans')
    visualize_retrieval(jeans, 10)

