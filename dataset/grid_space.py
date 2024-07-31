import math


class GridSpace:
    def __init__(self, x_unit: int, y_unit: int, x_min, y_min, x_max, y_max):
        assert x_unit > 0 and y_unit > 0

        self.x_unit = x_unit * 360 / (2 * math.pi * 6378137 * math.cos((y_min + y_max) * math.pi / 360))
        self.y_unit = y_unit * 360 / (2 * math.pi * 6378137)

        # whole space MBR range
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.x_size = int(math.ceil((x_max - x_min) / self.x_unit))
        self.y_size = int(math.ceil((y_max - y_min) / self.y_unit))
        self.grid_num = self.x_size * self.y_size

    def get_mbr(self, i_x, i_y):
        return self.x_min + self.x_unit * i_x, \
            self.y_min + self.y_unit * i_y, \
            self.x_min + self.x_unit * i_x + self.x_unit, \
            self.y_min + self.y_unit * i_y + self.y_unit

    def get_center_point(self, i_x: int, i_y: int):
        return self.x_min + self.x_unit / 2 + self.x_unit * i_x, \
            self.y_min + self.y_unit / 2 + self.y_unit * i_y

    def get_gridid_by_xyidx(self, i_x: int, i_y: int):
        return i_x * self.y_size + i_y

    def get_xyidx_by_gridid(self, grid_id: int):
        return grid_id // self.y_size, grid_id % self.y_size

    def get_gridid_range(self):
        return 0, self.x_size * self.y_size - 1

    def size(self):
        return self.x_size * self.y_size - 1

    def get_xyidx_by_point(self, x, y):
        assert self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
        i_x = int(math.floor((x - self.x_min) / self.x_unit))
        i_y = int(math.floor((y - self.y_min) / self.y_unit))

        return (i_x, i_y)

    def get_gridid_by_point(self, x, y):
        i_x, i_y = self.get_xyidx_by_point(x, y)
        return self.get_gridid_by_xyidx(i_x, i_y)

    def neighbour_gridids(self, i_x, i_y):
        # 8 neighbours
        x_r = [i_x - 1, i_x, i_x + 1]
        y_r = [i_y - 1, i_y, i_y + 1]
        x_r = list(filter(lambda x: 0 <= x < self.x_size, x_r))
        y_r = list(filter(lambda y: 0 <= y < self.y_size, y_r))

        xs = [l for l in x_r for _ in range(len(y_r))]
        ys = y_r * len(x_r)
        neighbours = zip(xs, ys)
        neighbours = filter(lambda xy: not (xy[0] == i_x and xy[1] == i_y), neighbours)

        return list(neighbours)

    def __str__(self):
        return "unit=({},{}), xrange=({},{}), yrange=({},{}), size=({},{})".format(
            self.x_unit, self.y_unit, self.x_min, self.x_max, self.y_min, self.y_max,
            self.x_size, self.y_size)
