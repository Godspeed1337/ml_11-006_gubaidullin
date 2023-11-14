import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict

from matplotlib import cm

from sklearn import svm


# Задаем случайное начальное значение для воспроизводимости
np.random.seed(42)


def generate_data_for_test(default=True):
    # Генерируем случайные данные для примера
    num_samples = 1000

    # Генерируем случайные признаки для класса 1
    class1 = np.random.randn(num_samples // 2, 2) + [1, 2]

    # Генерируем случайные признаки для класса -1
    class2 = np.random.randn(num_samples // 2, 2) + [4, 5]

    # Объединяем данные и метки классов
    x = np.vstack((class1, class2))
    y = np.hstack((np.ones(num_samples // 2), -1 * np.ones(num_samples // 2)))

    if default:
        x = np.array([[1, 2], [2, 3], [3, 3.5], [5, 4], [4, 2]])
        y = np.array([1, -1, -1, 1, -1])
    return x, y


def prepare_data_for_clf(first, second):
    new = first + second
    for i in range(len(new)):
        new[i] = [new[i][0], new[i][-1] * -1]
    inner_x = np.array(new)
    inner_y = np.hstack((np.ones(len(first)), -1 * np.ones(len(second))))
    return inner_x, inner_y


def create_clf(x, y):
    # Создаем объект SVM с линейным ядром
    inner_classifier = svm.SVC(kernel='linear')

    # Обучаем модель SVM
    inner_classifier.fit(x, y)

    # Получаем веса и смещение оптимальной гиперплоскости
    w = inner_classifier.coef_[0]
    b = inner_classifier.intercept_[0]

    # Визуализация данных и гиперплоскости
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Создаем сетку для построения гиперплоскости
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = inner_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Отображаем гиперплоскость и опорные векторы
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.scatter(inner_classifier.support_vectors_[:, 0], inner_classifier.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')
    plt.title('SVM Decision Boundary with Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    return inner_classifier


# Функция для рисования гиперплоскости в Pygame
def draw_hyperplane(screen, classifier, scale_factor: int = 100, screen_width: int = 600, screen_height: int = 400):
    # Получаем веса и смещение оптимальной гиперплоскости
    w = classifier.coef_[0]
    b = classifier.intercept_[0]

    # Рисуем гиперплоскость в виде прямой
    pygame.draw.line(screen, (255, 0, 0), (0, int(-b/w[1]*scale_factor + screen_height/2)),
                     (screen_width, int((-b-w[0]*screen_width)/w[1]*scale_factor + screen_height/2)))


def draw_hyperplane(inner_screen, inner_clf, inner_x, inner_y):
    w, b = inner_clf.coef_[0], inner_clf.intercept_[0]
    inner_scale_factor = 1
    x_min, x_max = np.min(inner_x[:, 0]), np.max(inner_x[:, 0])

    x1 = x_min
    y1 = (w[0] * x1 + b) / w[1]

    x2 = x_max
    y2 = (w[0] * x2 + b) / w[1]

    pygame.draw.line(inner_screen, (0, 0, 0),
                     (int(x1 * inner_scale_factor), int(y1 * inner_scale_factor)),
                     (int(x2 * inner_scale_factor), int(y2 * inner_scale_factor)), 2)

def dist(point_a, point_b):
    return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[-1] - point_b[-1]) ** 2)


def generate_random_points(coordinates):
    count = random.randint(2, 5)
    generated_points = list()
    for i in range(count):
        angle = np.pi * random.randint(0, 360) / 180
        radius = random.randint(10, 20)
        x = radius * np.cos(angle) + coordinates[0]
        y = radius * np.sin(angle) + coordinates[1]
        generated_points.append((x, y))
    return generated_points


def redraw_all_points(screen, f_points, s_points):
    screen.fill(color="#FFFFFF")
    for point in f_points:
        pygame.draw.circle(screen, color="#FF0000",
                           center=point, radius=r)
    for point in s_points:
        pygame.draw.circle(screen, color="#00FF00",
                           center=point, radius=r)


if __name__ == "__main__":
    r = 3
    epsilon = 25
    amount = 3
    screen_width: int = 600
    screen_height: int = 400
    scale_factor: int = 100

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    screen.fill(color="#FFFFFF")
    pygame.display.update()
    is_pressed_lmb = False
    is_pressed_rmb = False
    is_pressed_center = False
    first_points = []
    second_points = []
    running = True
    curr_color = None
    curr_points = None
    clf = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.VIDEORESIZE:
                redraw_all_points(screen, first_points, second_points)

            print(event)
            if event.type == pygame.KEYUP:
                if event.key == 13:
                    screen.fill(color="#FFFFFF")
                    first_points = []
                elif event.key == 27:
                    running = False
                elif event.key == 32:
                    redraw_all_points(screen, first_points, second_points)
                    x, y = prepare_data_for_clf(first_points, second_points)
                    clf = create_clf(x, y)
                    draw_hyperplane(screen, clf, x, y)
                    # draw_hyperplane(screen, clf,
                    #                 scale_factor=scale_factor, screen_width=screen_width, screen_height=screen_height)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_pressed_lmb = True
                elif event.button == 2:
                    is_pressed_center = True
                elif event.button == 3:
                    is_pressed_rmb = True
            if event.type == pygame.MOUSEBUTTONUP:
                is_pressed_lmb = False
                is_pressed_center = False
                is_pressed_rmb = False
                curr_color = None
                curr_points = None

            if is_pressed_lmb:
                curr_color = "#FF0000"
                curr_points = first_points
            elif is_pressed_rmb:
                curr_color = "#00FF00"
                curr_points = second_points
            elif is_pressed_center and clf is not None:
                coord = event.pos
                pred = clf.predict([[coord[0], coord[-1] * -1]])
                if pred[0] == 1:
                    first_points.append(coord)
                    pygame.draw.circle(screen, color="#FF0000", center=coord, radius=r)
                else:
                    second_points.append(coord)
                    pygame.draw.circle(screen, color="#00FF00", center=coord, radius=r)
            
            if (is_pressed_lmb or is_pressed_rmb) and (curr_points is not None and curr_color is not None):
                coord = event.pos
                if len(curr_points):
                    if dist(curr_points[-1], coord) > 5 * r:
                        pygame.draw.circle(screen, color=curr_color, center=coord, radius=r)
                        near_point = generate_random_points(coord)
                        curr_points.extend(near_point)
                        for elem in near_point:
                            pygame.draw.circle(screen, color=curr_color, center=elem, radius=r)
                        curr_points.append(coord)
                else:
                    pygame.draw.circle(screen, color=curr_color, center=coord, radius=r)
                    curr_points.append(coord)

            pygame.display.update()

    pygame.quit()
