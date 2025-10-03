class WCL:
    def __init__(self, weight, coordinate):
        self.weight = weight
        self.coordinate = coordinate

    def calculate_coordinate(self):
        x = sum(w * coord[0] for w, coord in zip(self.weight, self.coordinate)) / sum(
            self.weight
        )
        y = sum(w * coord[1] for w, coord in zip(self.weight, self.coordinate)) / sum(
            self.weight
        )
        T = (x, y)

        return T
