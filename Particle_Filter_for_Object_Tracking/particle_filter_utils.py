import numpy as np
from PIL import Image, ImageDraw
from skimage import color


class Template:
    def __init__(self, img, x, y, w, h) -> None:
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.model = img[self.y : self.y + self.h, self.x : self.x + self.w, ...]


def visualize_particle_filter(img, particles, state, template):
    # if img.ndim == 2:
    #     img = color.gray2rgb(img)
    #     img = float_to_uint8(img)
    if template.model.ndim == 2:
        img[0 : template.h, 0 : template.w, :] = color.gray2rgb(template.model)
    elif template.model.ndim == 3:
        img[0 : template.h, 0 : template.w, :] = template.model

    x = int(state[0])
    y = int(state[1])
    w = int(state[2])
    h = int(state[3])

    pillow_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pillow_img)
    draw_particles(draw, particles)
    draw_window(draw, x - w / 2, y - h / 2, w, h)

    return np.asarray(pillow_img)


def draw_particles(draw, particles):
    radius = 1.5
    for p in particles:
        x = p[0]
        y = p[1]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 255, 0),
            width=2,
        )


def draw_window(draw, x, y, w, h):
    x1, y1 = x + w, y + h
    draw.rectangle([(x, y), (x1, y1)], outline=(0, 255, 0))
