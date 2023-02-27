import cv2 as cv
import numpy as np

cap = cv.VideoCapture('videos/video2.mp4')
oblako1 = cv.imread('videos/oblako.jpg',)
oblako = cv.resize(oblako1, [640, 360])

#размеры блока
y_step = 15
x_step = 15
#col_reg = 0
# Rазмер поиска

size = 3
frames = 2
# первый и предыдущий кадр
suc, frame = cap.read()
prev_frame = frame[:]
while True:
    # задаём предыдущий кадр
    prev_frame = frame[:]

    suc, frame = cap.read()  # покадрово считываем видео если suc = True
    suc, original = cap.read()
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    arr = np.asarray(gray_image, dtype='uint8')


    def img_fil(image):

        image = cv.bilateralFilter(gray_image, 10, 160, 15)
        return image


    edges = cv.Canny(img_fil(frame), 30, 50) # переделать как сделаю функцию обработки
    ret, tresh = cv.threshold(gray_image, 179, 255, cv.THRESH_BINARY)
    cont = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    def bright_points_to_green(image):

        # находим точки с яркостью N и выше
        bright_points = gray_image > 179 # 190 интересный вариант

        # Точки с яркостью 179 и выше меняем на зелёные
        frame[bright_points] = [0, 255, 0] #image or frame

        return image


    def regions(image,):

        # Обход регионов
        redjions = []

        for y in range(0, image.shape[0], y_step):
            for x in range(0, image.shape[1],  x_step):

                region = image[y:y + y_step,  x:x + x_step]
                redjions.append(region)

        lens = len(redjions)
        #print(lens)
        return redjions #redjions image


    def get_coordinates(image, x_step, y_step):
        cords = []

        for y in range(0, image.shape[0], y_step):
            for x in range(0, image.shape[1], x_step):
                region_start = (x, y)
                region_end = (x + x_step, y + y_step)
                cords.append((region_start, region_end))

        return cords  # возвращает список координат регионов


    def draw_mask(image, regions, x_step, y_step):
        mask = image.copy()
        coords = get_coordinates(image, x_step, y_step)

        for region, (start, end) in zip(regions, coords):
            cv.rectangle(mask, start, end, (0, 0, 255), 2)

        return mask  # возвращает маску регионов


    def mask(image, regions, coords):

        for region, (x1, y1), (x2, y2) in zip(regions, coords):
            for y in range(region.shape[0]):
                for x in range(region.shape[1]):
                    if region[y, x] == [0, 255, 0]:
                        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        return image


    def SumAbsDiff(image, prev_image):
        # Проверяем входные данные
        if len(image.shape) != 2:
            raise Exception('Функция принимает только двумерные изображения')

        # Сумма абсолютных отличий
        sum_diff = 0
        # Проходим по всем пикселям изображения, переделать в пикселы(Для этого переделать regions)
        for x in range(image.shape[0],  x_step):
            for y in range(image.shape[1],  y_step):
                # Забираем яркость пикселей
                cur_pixel = image[x,  y]
                prev_pixel = prev_image[x,  y]
                #sum_diff += abs(image[x,  y] - prev_image[x,  y])
                        # Вычисляем разность яркостей и добавляем ее к общей сумме
                sum_diff += abs(cur_pixel - prev_pixel)

        return sum_diff


    def sad(region_a, region_b):
        # Сумма квадратов разностей яркостей пикселов
        ssd = np.sum((region_a - region_b) ** 2)

        return ssd  # возвращает сумму квадратов разностей яркостей пикселов

    def find_motion(image1, image2, x_step, y_step, size):
        regions_1 = regions(image1, x_step, y_step)
        regions_2 = regions(image2, x_step, y_step)

        coords_1 = get_coordinates(image1, x_step, y_step)
        coords_2 = get_coordinates(image2, x_step, y_step)

        motions = []

        for i in range(len(regions_1)):
            region_1 = regions_1[i]
            region_2 = regions_2[i]

            ssd = sad(region_1, region_2)

            if ssd > 0:
                motion = coords_2[i][0] - coords_1[i][0], coords_2[i][1] - coords_1[i][1]
                motions.append(motion)

        return motions  # возвращает список движений в регионах


    img = bright_points_to_green(gray_image)
    reg1 = regions(frame)

    draw = draw_mask(original, reg1, x_step, y_step)

    cv.imshow('obl', draw)  # edges )
    cv.imshow('obl2', original) # reg1
    cv.imshow('obl1', prev_frame) #original


    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.waitKey(0)