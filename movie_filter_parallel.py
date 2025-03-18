# Ce programme va charger n images et y appliquer un filtre de netteté en parallèle
# en utilisant un modèle maître-esclave avec MPI

from PIL import Image
import os
import numpy as np
from scipy import signal
import time
import csv
from datetime import datetime
import cProfile
import functools
from mpi4py import MPI

PATH = "datas/perroquets/"
# On crée un dossier de sortie
if not os.path.exists("sorties/perroquets"):
    os.makedirs("sorties/perroquets")
OUT_PATH = "sorties/perroquets/"

# Constantes pour la taille des images
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Nom du fichier CSV global
CSV_FILENAME = "image_processing_parallel_results.csv"

def apply_filter(image):
    """
    Applique le filtre de netteté à une image
    """
    # On charge l'image
    img = Image.open(PATH + image)
    print(f"Taille originale {img.size}")
    # Conversion en HSV :
    img = img.convert('HSV')
    # On convertit l'image en tableau numpy et on normalise
    img = np.repeat(np.repeat(np.array(img), 2, axis=0), 2, axis=1)
    img = np.array(img, dtype=np.double)/255.
    print(f"Nouvelle taille : {img.shape}")
    # Tout d'abord, on crée un masque de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On applique le filtre de flou
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:,:,i] = signal.convolve2d(img[:,:,i], mask, mode='same')
    # On crée un masque de netteté
    mask = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
    # On applique le filtre de netteté
    sharpen_image = np.zeros_like(img)
    sharpen_image[:,:,:2] = blur_image[:,:,:2]
    sharpen_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask, mode='same'), 0., 1.)

    sharpen_image *= 255.
    sharpen_image = sharpen_image.astype(np.uint8)
    # On retourne l'image modifiée
    return Image.fromarray(sharpen_image, 'HSV').convert('RGB')

def master_process(comm, nb_images):
    """
    Processus maître qui distribue les tâches
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    counter = 1
    filenames = ["Perroquet{:04d}.jpg".format(i+1) for i in range(nb_images)]
    while (counter < nb_images):
        for i in range(1, size):
            if comm.Iprobe(source=i, tag=i):
                _ = comm.recv(source=i, tag=i)
                if counter < nb_images:
                    comm.send(filenames[counter-1], dest=i, tag=i)
                    counter += 1
                else:
                    comm.send(None, dest=i, tag=i)


def slave_process(comm):
    """
    Processus esclave qui traite les images
    """
    while True:
        rank = comm.Get_rank()
        comm.send(None, dest=0, tag=rank) #on indique au master_process qu'on est prêt à recevoir
        filename = comm.recv(source=0, tag=rank)
        
        if filename is None:
            break
        else:
            img = apply_filter(filename)
            img.save(OUT_PATH + filename)
        
        #si on enregistre les images, on le ferait ici

def test_time(nb_images):
    """
    Décorateur pour tester les performances de traitement d'images en parallèle
    Args:
        nb_images: Nombre d'images à traiter
    """
    def profile(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            
            
            if rank == 0:
                start_time = time.perf_counter()
            
            if rank == 0:
                results = master_process(comm, nb_images)
            else:
                slave_process(comm)

            if rank == 0:
                total_time = time.perf_counter() - start_time
                
                file_exists = os.path.exists(CSV_FILENAME)
                
                with open(CSV_FILENAME, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    
                    if not file_exists:
                        writer.writerow(["timestamp", "nb_images", "nb_processes", 
                                       "total_time (s)", "time_per_image (s)"])
                    
                    time_per_image = total_time / nb_images
                    
                    writer.writerow([
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        nb_images,
                        size,
                        f"{total_time:.3f}",
                        f"{time_per_image:.3f}"
                    ])
                
                print(f"Traitement parallèle de {nb_images} images {IMAGE_WIDTH}x{IMAGE_HEIGHT}:")
                print(f"  Nombre de processus: {size}")
                print(f"  Temps total: {total_time:.3f}s")
                print(f"  Temps moyen par image: {time_per_image:.3f}s")
                print(f"Résultats ajoutés dans {CSV_FILENAME}")
            
            return results if rank == 0 else None

        return wrapper
    return profile

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    test_nb_images = [1, 5, 10, 20, 37]
    
    if rank == 0:
        total_start_time = time.perf_counter()
    
    for nb_images in test_nb_images:
        @test_time(nb_images)
        def process_images(nb_images):
            return None  # Le traitement est fait dans master_process
        
        process_images(nb_images)
    
    if rank == 0:
        total_time = time.perf_counter() - total_start_time
        print(f"\nTemps total d'exécution de tous les tests: {total_time:.3f}s")
    
    MPI.Finalize() 