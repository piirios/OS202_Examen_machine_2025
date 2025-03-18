from PIL import Image
import os
import numpy as np
from scipy import signal
from mpi4py import MPI
import time
import csv
from datetime import datetime
from tqdm import tqdm

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
CHUNK_SIZE = 3  # Taille minimale pour le filtre 3x3


CSV_FILENAME = "double_size_performance.csv"

def process_chunk(chunk_data):
    """
    Traite un chunk d'image avec les filtres
    """
    # On crée un masque de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    
    # On applique le filtre de flou
    blur_chunk = np.zeros_like(chunk_data, dtype=np.double)
    for i in range(3):
        blur_chunk[:,:,i] = signal.convolve2d(chunk_data[:,:,i], mask, mode='same')
    
    # On crée un masque de netteté
    mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    # On applique le filtre de netteté uniquement sur la luminance
    res = np.zeros_like(chunk_data, dtype=np.double)
    res[:,:,:2] = blur_chunk[:,:,:2]
    res[:,:,2] = np.clip(signal.convolve2d(blur_chunk[:,:,2], mask, mode='same'), 0., 1.)
    
    return res

def double_size(image):
    """
    Double la taille d'une image en parallélisant le traitement avec des blocs de 3x3
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        img = Image.open(image)
        print(f"Taille originale {img.size}")
        img = img.convert('HSV')
        img = np.repeat(np.repeat(np.array(img, dtype=np.double), 2, axis=0), 2, axis=1)/255.
        print(f"Nouvelle taille : {img.shape}")
        print(f"Création et distribution des chunks...")
        
        # Diviser l'image en blocs de 3x3 et les distribuer directement
        total_chunks = img.shape[0] * img.shape[1]
        chunk_count = 0
        chunk_data = None  # Pour stocker le chunk du processus 0
        
        # Barre de progression pour la création et distribution des chunks
        pbar = tqdm(total=total_chunks, desc="Création et distribution des chunks")
        
        for i in range(-1, img.shape[0]-2):
            for j in range(-1, img.shape[1]-2):
                #(i,j) est le coin haut gauche du chunk
                # Créer un chunk de 3x3
                chunk = np.zeros((3, 3, 3), dtype=np.double)
                
                if (i == -1):
                    if (j == -1):
                        print("")
                    elif (j== img.shape[1]-3):
                        print("")
                    else:
                        print("")
                elif (i == img.shape[0]-3):
                    if (j == -1):
                        print("")
                    elif (j== img.shape[1]-3):
                        print("")
                    else:
                        print("")
                else:
                    if (j == -1):
                        #coté gauche, hors coin
                        chunk[:,2:3,:] = img[i:i+3,0:1,: ]
                        chunk[:,1,:] = img[i:i+3,0,: ]
                    elif (j== img.shape[1]-3):
                        #coté droit, hors coin
                        chunk[:,:2,:] = img[i:i+3,j:j+2,: ]
                        chunk[:,3,:] = img[i:i+3,j+2,: ]
                    else:
                        #cas normal
                        chunk[:,:,:] = img[i:i+3,j:j+3,: ]

                # Distribuer directement le chunk
                dest = chunk_count % size
                if dest == 0:
                    chunk_data = chunk
                else:
                    comm.send(chunk, dest=dest, tag=0)
                
                chunk_count += 1
                pbar.update(1)
        
        pbar.close()
    else:
        chunk_data = comm.recv(source=0, tag=0)
    
    if rank == 0:
        print(f"Traitement des chunks par le processus {rank}...")
    processed_chunk = process_chunk(chunk_data)
    
    if rank == 0:
        print("Reconstruction de l'image...")
        # Créer l'image résultat
        result = np.zeros_like(img)
        chunk_idx = 0
        
        # Barre de progression pour la reconstruction
        pbar = tqdm(total=total_chunks, desc="Reconstruction de l'image")
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if chunk_idx % size == 0:
                    chunk = processed_chunk
                else:
                    chunk = comm.recv(source=chunk_idx % size, tag=1)
                result[i,j,:] = chunk[1,1,:]  # On ne prend que le pixel central
                chunk_idx += 1
                pbar.update(1)
        
        pbar.close()
        
        result = (255.*result).astype(np.uint8)
        print("Conversion finale de l'image...")
        return Image.fromarray(result, 'HSV').convert('RGB')
    else:
        print(f"Envoi des résultats du processus {rank} au processus 0...")
        comm.send(processed_chunk, dest=0, tag=1)
        return None

def test_performance(image):
    """
    Teste les performances pour différents nombres de processus
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        file_exists = os.path.exists(CSV_FILENAME)
        
        with open(CSV_FILENAME, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            if not file_exists:
                writer.writerow(["timestamp", "nb_processes", 
                               "total_time (s)", "processing_time (s)", 
                               "communication_time (s)"])
            
            start_time = time.perf_counter()
            
            process_start = time.perf_counter()
            doubled_image = double_size(image)
            process_time = time.perf_counter() - process_start
            
            total_time = time.perf_counter() - start_time
            comm_time = total_time - process_time
            
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                comm.Get_size(),
                f"{total_time:.3f}",
                f"{process_time:.3f}",
                f"{comm_time:.3f}"
            ])
            
            print(f"Test avec {comm.Get_size()} processus:")
            print(f"  Temps total: {total_time:.3f}s")
            print(f"  Temps de traitement: {process_time:.3f}s")
            print(f"  Temps de communication: {comm_time:.3f}s")
            
            doubled_image.save(f"sorties/paysage_double_{comm.Get_size()}processes.jpg")
        
        print(f"Résultats sauvegardés dans {CSV_FILENAME}")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        total_start_time = time.perf_counter()
        
        path = "datas/"
        image = path+"paysage.jpg"
        test_performance(image)
        
        total_time = time.perf_counter() - total_start_time
        print(f"\nTemps total d'exécution: {total_time:.3f}s")
    
    MPI.Finalize()
