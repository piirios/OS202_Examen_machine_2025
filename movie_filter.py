# Ce programme va charger n images et y appliquer un filtre de netteté
# puis les sauvegarder dans un dossier de sortie

from PIL import Image
import os
import numpy as np
from scipy import signal
import time
import csv
from datetime import datetime
import cProfile
import functools

PATH = "datas/perroquets/"
# On crée un dossier de sortie
if not os.path.exists("sorties/perroquets"):
    os.makedirs("sorties/perroquets")
OUT_PATH = "sorties/perroquets/"

# Constantes pour la taille des images
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Nom du fichier CSV global
CSV_FILENAME = "image_processing_results.csv"

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

def test_time(nb_images):
    """
    Décorateur pour tester les performances de traitement d'images
    Args:
        nb_images: Nombre d'images à traiter
    """
    def profile(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Création ou ouverture du fichier CSV
            file_exists = os.path.exists(CSV_FILENAME)
            
            with open(CSV_FILENAME, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                
                # Écriture de l'en-tête si le fichier est nouveau
                if not file_exists:
                    writer.writerow(["timestamp", "nb_images", "total_time (s)", 
                                   "processing_time (s)", "io_time (s)", 
                                   "time_per_image (s)"])

                # Liste des fichiers à traiter
                filenames = ["Perroquet{:04d}.jpg".format(i+1) for i in range(nb_images)]
                
                # Mesure du temps total
                start_time = time.perf_counter()
                
                # Mesure du temps de traitement
                process_start = time.perf_counter()
                for filename in filenames:
                    result = func(filename)
                    result.save(OUT_PATH + filename)
                process_time = time.perf_counter() - process_start
                
                # Temps total
                total_time = time.perf_counter() - start_time
                
                # Temps I/O
                io_time = total_time - process_time
                
                # Temps par image
                time_per_image = total_time / nb_images

                # Écriture des résultats
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    nb_images,
                    f"{total_time:.3f}",
                    f"{process_time:.3f}",
                    f"{io_time:.3f}",
                    f"{time_per_image:.3f}"
                ])

                print(f"Traitement de {nb_images} images {IMAGE_WIDTH}x{IMAGE_HEIGHT}:")
                print(f"  Temps total: {total_time:.3f}s")
                print(f"  Temps moyen par image: {time_per_image:.3f}s")
                print(f"  Temps de traitement: {process_time:.3f}s")
                print(f"  Temps I/O: {io_time:.3f}s")

            print(f"Résultats ajoutés dans {CSV_FILENAME}")
            return result

        return wrapper
    return profile

if __name__ == "__main__":
    # Test avec différentes quantités d'images
    test_nb_images = [1, 5, 10, 20, 37]
    
    # Mesure du temps total d'exécution de tous les tests
    total_start_time = time.perf_counter()
    
    for nb_images in test_nb_images:
        @test_time(nb_images)
        def process_image(filename):
            return apply_filter(filename)
        
        # Exécution des tests
        process_image("Perroquet0001.jpg")
    
    total_time = time.perf_counter() - total_start_time
    print(f"\nTemps total d'exécution de tous les tests: {total_time:.3f}s")
    MPI.Finalize()

