import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

plt.style.use('default')
sns.set_palette("husl")

def load_and_process_data():
    csv_files = glob.glob("image_processing_parallel_results*.csv")
    all_data = []
    
    for file in csv_files:
        parts = file.split('_')
        if len(parts) == 4:  # Fichier sans numéro de processus
            nb_processes = 1
        else:
            nb_processes = int(parts[-1].split('.')[0])
        df = pd.read_csv(file)
        df['nb_processes'] = nb_processes
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def plot_performance_curves(df):
    plt.figure(figsize=(12, 6))
    for nb_proc in df['nb_processes'].unique():
        data = df[df['nb_processes'] == nb_proc]
        plt.plot(data['nb_images'], data['total_time (s)'], 
                marker='o', label=f'{nb_proc} processus')
    
    plt.xlabel('Nombre d\'images')
    plt.ylabel('Temps total (s)')
    plt.title('Temps total vs nombre d\'images')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_total_time.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    base_time = df[df['nb_processes'] == 1].set_index('nb_images')['total_time (s)']
    
    for nb_proc in df['nb_processes'].unique():
        if nb_proc > 1:
            data = df[df['nb_processes'] == nb_proc]
            speedup = base_time / data.set_index('nb_images')['total_time (s)']
            plt.plot(speedup.index, speedup.values, 
                    marker='o', label=f'{nb_proc} processus')
    
    plt.xlabel('Nombre d\'images')
    plt.ylabel('Speed-up')
    plt.title('Speed-up vs nombre d\'images')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_speedup.png')
    plt.close()

def calculate_speedup_table(df):
    # Calculer le speed-up moyen pour chaque nombre de processus
    base_time = df[df['nb_processes'] == 1].set_index('nb_images')['total_time (s)']
    speedups = []
    
    for nb_proc in sorted(df['nb_processes'].unique()):
        if nb_proc > 1:
            data = df[df['nb_processes'] == nb_proc]
            speedup = base_time / data.set_index('nb_images')['total_time (s)']
            # Remplacer les valeurs infinies par NaN et calculer la moyenne
            speedup = speedup.replace([np.inf, -np.inf], np.nan)
            mean_speedup = speedup.mean()
            if np.isnan(mean_speedup):
                mean_speedup = "N/A"
            else:
                mean_speedup = f"{mean_speedup:.2f}x"
            speedups.append((nb_proc, mean_speedup))
    
    return speedups

def main():
    df = load_and_process_data()
    
    plot_performance_curves(df)
    
    speedups = calculate_speedup_table(df)
    
    markdown_table = "## Speed-up en fonction du nombre de processus\n\n"
    markdown_table += "| Nombre de processus | Speed-up moyen |\n"
    markdown_table += "|-------------------|----------------|\n"
    for nb_proc, speedup in speedups:
        markdown_table += f"| {nb_proc} | {speedup} |\n"
    
    with open('speedup_table.md', 'w') as f:
        f.write(markdown_table)

if __name__ == "__main__":
    main() 