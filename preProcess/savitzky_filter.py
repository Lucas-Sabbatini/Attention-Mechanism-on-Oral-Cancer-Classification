import numpy as np
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.pyplot as plt


class SavitzkyFilter:
    def __init__(self, window_length=11, poly_order=3):
        self.window_length = window_length  
        self.poly_order = poly_order

    def buildFilter(self, x : np.ndarray):
        return savgol_filter(x,self.window_length,self.poly_order)
    
    def plotFilter(self,x_raw :np.ndarray, x_filtered :np.ndarray):
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=range(len(x_raw)), y=x_raw, label="Original", color="red")
        sns.lineplot(x=range(len(x_filtered)), y=x_filtered, label="Filtrado", color="blue")
        plt.title("Filtro Savitzky-Golay")
        plt.xlabel("Valor")
        plt.ylabel("Valores de ativação")
        plt.legend()
        plt.savefig("grafico.png", dpi=300, bbox_inches="tight")
    

if __name__ == '__main__':
    # Sinal com ruído
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, 100)

    # Aplicar filtro
    filtro = SavitzkyFilter(window_length=11, poly_order=3)
    y_filtered = filtro.buildFilter(y)

    # Plotar
    filtro.plotFilter(y, y_filtered)