__authors__ = ['1679933','1689435']
__group__ = '100'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, visualize_k_means, Plot3DCloud
import utils
import KNN
import Kmeans
import numpy as np
import time
import matplotlib.pyplot as plt


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
        # crear llista que contindrà indexs de coincidents
        indexs = []
        # iterar per les etiquetes de colors i imatges per trobar coincidència amb query_color
        for i, (img, label) in enumerate(zip(images, labels)):
            # verificar si els colors de l'etiqueta coincideixen amb alguna query_color, np.char.lower() converteix a minúscules
            # per no tenir errors en la comparació, any() retorna True si algun dels elements de query_color és igual
            # que l'etiqueta
            if any(query in np.char.lower(label) for query in np.char.lower(query_color)):
                # si es troba imatge que coincideix en colors amb query_color, s'afegeix a retrieved_images[]
                retrieved_images.append(img)
                indexs.append(i)
        # es retorna retrieves_images[], que ja conté totes les imatges coincidents en color amb query_color
        return indexs, np.array(retrieved_images)


    def retrieval_by_shape(images, labels, query_shape):
        # crear llista que contindrà les imatges que coincideixin en colors
        retrieved_images = []
        # crear llista que contindrà indexs de coincidents
        indexs = []
        # iterar per les etiquetes de forma i imatges per trobar coincidència amb query_shape
        for i, (img, label) in enumerate(zip(images, labels)):
            # verificar si la forma de l'etiqueta coincideix amb query_shape, .lower() converteix a minúscules
            # per no tenir errors en la comparació
            if query_shape.lower() == label.lower():
                # si es troba imatge que coincideix en forma amb query_shape, s'afegeix a retrieved_images[]
                retrieved_images.append(img)
                indexs.append(i)
        # es retorna retrieves_images[], que ja conté totes les imatges coincidents en forma amb query_shape
        return indexs, np.array(retrieved_images)


    def retrieval_combined(images, color_labels, shape_labels, query_color, query_shape):
        # si query_color no és una llista (és un sol string ja que el query és d'un sol color), es converteix a llista
        # per poder-la iterar
        if type(query_color) != list:
            query_color = [query_color]

        # crear llista que contindrà les imatges que coincideixin
        retrieved_images = []
        # crear llista que contindrà indexs de coincidents
        indexs = []
        # iterar per les llistes d'imatges, etiquetes de colors i etiquetes de formes a la vegada
        for i, (img, color_label, shape_label) in enumerate(zip(images, color_labels, shape_labels)):
            # comprovar si l'etiqueta de forma d'aquella imatge coincideix amb la query_shape
            if query_shape.lower() == shape_label.lower():
                # si coincideix en forma, comprovar si alguna de les query_color és l'etiqueta de color
                if any(query in np.char.lower(color_label) for query in np.char.lower(query_color)):
                    # si coincideix tant en forma com en color amb query_shape i query_color, afegir a
                    # retrieved_images[]
                    retrieved_images.append(img)
                    indexs.append(i)
        # es retorna retrieves_images[], que ja conté totes les imatges coincidents en forma i color
        return indexs, np.array(retrieved_images)

    def kmean_statistics(kmeans, kmax):
        iter = []
        wcd = []
        time_list = []
        for k in range(2, kmax + 1):
            kmeans.K = k
            start = time.time()
            kmeans.fit()
            end = time.time()
            iter.append(kmeans.num_iter)
            wcd.append(kmeans.withinClassDistance())
            time_list.append(end - start)
        
        return wcd, iter, time_list

    def visualize_statistics(iter, wcd, time_list, kmax):
        total_iter = range(2, kmax + 1)
        plt.figure(figsize=(10, 6))
    
        plt.subplot(1, 3, 1)
        plt.plot(total_iter, wcd, marker='o')
        plt.title('Distància intra-clas (WCD)')
        plt.xlabel('Número de clústers K')
        plt.ylabel('WCD')
        
        plt.subplot(1, 3, 2)
        plt.plot(total_iter, iter, marker='o')
        plt.title('Número de iteracions')
        plt.xlabel('Número de clústers (K)')
        plt.ylabel('Iteracions')
        
        plt.subplot(1, 3, 3)
        plt.plot(total_iter, time_list, marker='o')
        plt.title('Temps de convergència')
        plt.xlabel('Número de clústers (K)')
        plt.ylabel('Temps (segons)')
        
        plt.tight_layout()
        plt.show()

    def get_shape_accuracy(labels, ground_truth):
        labels_len = len(labels)
        if labels_len != len(ground_truth):
            return None

        correct_shapes = np.equal(labels, ground_truth)
        accuracy_percentage = np.count_nonzero(correct_shapes) / labels_len * 100

        return accuracy_percentage

    def get_color_accuracy():
        pass



    # QUALITATIVE ANALYSIS

    # input

    # 0: Kmeans (retrieval by color),
    # 1: KNN (retrieval by shape),
    # 2: Kmeans and KNN combined (retrieval by color and shape)
    # 3: None
    my_type = 0
    # string o array de strings, case-insesitive
    # s'accepten més d'un color però només una forma
    # exemples:
    # 'blue'
    # ['pink', 'blue']
    # 'Flip flOPS'
    # ['pinK', 'jeAns']
    my_color_query = 'pink'
    my_shape_query = 'handbags'

    # set up
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

    # analysis
    if type(my_color_query) != list:
        my_color_query = [my_color_query]

    if my_type == 0:
        idx, color = retrieval_by_color(test_imgs, color_labels, my_color_query)
        truth = []
        truth_labels = []
        for i in idx:
            truth_labels.append(test_color_labels[i])
            truth.append(True if any(query in np.char.lower(test_color_labels[i]) for query in np.char.lower(my_color_query)) else False)
        visualize_retrieval(color, 20, info=truth_labels, ok=truth, title=my_color_query)

    elif my_type == 1:
        idx, shape = retrieval_by_shape(test_imgs, shape_labels, my_shape_query)
        truth = []
        truth_labels = []
        for i in idx:
            truth_labels.append(test_class_labels[i])
            truth.append(True if my_shape_query.lower() == test_class_labels[i].lower() else False)
        visualize_retrieval(shape, 20, info=truth_labels, ok=truth, title=my_shape_query)

    elif my_type == 2:
        idx, combined = retrieval_combined(test_imgs, color_labels, shape_labels, my_color_query, my_shape_query)
        truth = []
        truth_labels = []
        for i in idx:
            truth_labels.append([test_color_labels[i], test_class_labels[i]])
            truth.append(True if my_shape_query.lower() == test_class_labels[i].lower() and
                                 any(query in np.char.lower(test_color_labels[i]) for query in np.char.lower(my_color_query))
                         else False)
        visualize_retrieval(combined, 20, info=truth_labels, ok=truth, title=f"{my_color_query}, {my_shape_query}")

    elif my_type == 3:
        pass

    else:
        print('NOMÉS 0, 1, 2 o 3 recorxolis!')

    """

    #combined = retrieval_combined(test_imgs, color_labels, shape_labels, 'PInk', 'HandBAGs')
    #visualize_retrieval(combined, 20)

    # ------------- quantitative analysis ---------------

    #wcd, iter, time_list = kmean_statistics(km, 10)
    #visualize_statistics(wcd, iter, time_list, 10)

    #accuracy = get_shape_accuracy(shape_labels, test_class_labels)
    #print(f"Percentatge d'etiquetes correctes: {accuracy}%")
    """