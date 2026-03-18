import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Настройка визуализации
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 4)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Для корректного отображения кириллицы
plt.rcParams['font.family'] = 'DejaVu Sans'

# 1. Generate data for the x-axis
# Use numpy.linspace to create 100 evenly spaced points from 0 to 2*pi
x = np.linspace(-4, 8, 10000)
init_h = 2
init_k = 2

fig = plt.figure(figsize=(14, 3))
plt.subplots_adjust(bottom=0.25, top=0.95, hspace=0.4)


# 2. Calculate the sine of each x value
def f(x,h,k):
    return np.exp(-g(x,h,k)**2/2)
def f0(x):
    return np.exp(-x**2/2)

def g(x,h,k):
    return x/(1+(h-1)*s(x,k))

def s(x,k):
    return 1/(1+np.exp(-k*x))

# 3. Plot the data
line_g, = plt.plot(x, g(x, init_h, init_k), lw=2, label=r'$g(x)=x/(1+(h-1)s(x))$', color='#3498db')
line_s, = plt.plot(x, s(x, init_k), lw=2, label=r'$s(x)=1/(1+e^{-kx})$',         color='#9b59b6')
line_f, = plt.plot(x, f(x, init_h, init_k), lw=2, label=r'$f(x)=e^{-g(x)^2/2}$', color='#2ecc71')
plt.plot(x, f0(x), '--', linewidth=2,                           color='#000000')

# 4. Add labels and title for clarity
plt.xlabel("Отключение")
plt.ylabel("f(x)")
plt.title("")

plt.legend()

plt.ylim(-0.3,1.3)
plt.grid(True) # Add grid lines


# Создаём область для слайдеров (левая нижняя часть фигуры)
# axcolor = 'lightgoldenrodyellow'
# ax_h = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)   # [left, bottom, width, height]
# ax_k = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor=axcolor)

# # Создаём ползунки
# slider_h = Slider(ax_h, 'h', -10, 10, valinit=init_h, valstep=0.1)
# slider_k = Slider(ax_k, 'k', -10, 10, valinit=init_k, valstep=0.1)

# # Функция обновления графиков при изменении ползунков
# def update(val):
#     h = slider_h.val
#     k = slider_k.val
    
#     # Пересчитываем y-данные
#     line_f.set_ydata(f(x,h,k))
#     line_g.set_ydata(g(x,h,k))
#     line_s.set_ydata(s(x,k))
    
#     # Перерисовываем фигуру
#     fig.canvas.draw_idle()

# # Привязываем функцию к событиям изменения ползунков
# slider_h.on_changed(update)
# slider_k.on_changed(update)

# 5. Display the plot
plt.savefig(f'FULL/low_decreade_function.png', dpi=300, bbox_inches='tight', facecolor='white')
# plt.show()