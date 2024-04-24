import numpy as np

# returns [fraction of screen covered by circular viewport, fraction of screen covered by square viewport, side length (in px) of square viewport]
def calc_viewport(screen_resolution, screen_size, view_distance, viewport_angle=10):
  # find radius and area of conic base
  radius = view_distance * np.tan(viewport_angle * np.pi / 180)
  circle_area = np.pi * radius ** 2

  # find relative length of diagonal of screen
  aspect_ratio = screen_resolution // np.gcd(screen_resolution[0], screen_resolution[1])
  hypotenuse = (aspect_ratio[0]**2 + aspect_ratio[1]**2)**0.5

  # find dimensions of screen
  screen_dimensions = aspect_ratio / hypotenuse * screen_size
  screen_area = screen_dimensions[0] * screen_dimensions[1]

  # calculate pixel density and largest square inside circle
  ppcm = screen_resolution[0] / screen_dimensions[0]
  square_side_length = np.sqrt(2) * radius
  square_size_pixels = int(np.ceil(square_side_length * ppcm))

  return [circle_area / screen_area, square_side_length**2 / screen_area, square_size_pixels]