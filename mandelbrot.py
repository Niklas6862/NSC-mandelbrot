import numpy
import matplotlib.pyplot as plt
import time


def mandelbrot(width, height, max_iter,
               xmin, xmax, ymin, ymax):

    img = [[0] * width for _ in range(height)]

    for y in range(height):
        im = ymin + (y / (height - 1)) * (ymax - ymin)

        for x in range(width):
            re = xmin + (x / (width - 1)) * (xmax - xmin)
            c = complex(re, im)

            z = 0j
            n = 0

            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z * z + c
                n += 1

            img[y][x] = n

    return img


width = 1024
height = 1024
max_iter = 200
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5

start = time.time()

data = mandelbrot(width, height, max_iter, xmin, xmax, ymin, ymax)

end = time.time()
runtime = end - start

print("Runtime:", runtime, "seconds")

plt.imshow(data, extent=(xmin, xmax, ymin, ymax), origin="lower")
plt.title("Mandelbrot set (naive Python)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()

