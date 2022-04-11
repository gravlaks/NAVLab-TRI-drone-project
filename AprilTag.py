
import json
class AprilTag:
    def __init__(self, tag_id, testing_day="thunderhill_04_07"):
        filepath = "data/" + "measurements.json"
        with open(filepath, "r") as f:
            car = json.load(f)[testing_day][str(tag_id)]
            self.w = car["w"]
            self.h = car["h"]
            self.type = car["type"]

if __name__ == '__main__':
    atag = AprilTag(0)
    print(atag.w)