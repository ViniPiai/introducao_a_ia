import json
from datetime import datetime


class Register(object):
    name = ''
    tag = ''
    timestamp = 0
    date = datetime.now()
    x = 0.0
    y = 0.0
    z = 0.0
    activity = ''

    def __init__(self, row: list):
        self.name = row[0]
        self.tag = row[1]
        self.timestamp = row[2]
        self.date = self.convert_to_datetime(row[3])
        self.x = float(row[4])
        self.y = float(row[5])
        self.z = float(row[6])
        self.activity = row[7]

    def convert_to_datetime(self, value: str):
        date, time = value.split(" ")
        date = [int(x) for x in date.split(".")]
        time = [int(x) for x in time.split(":")]
        return datetime(date[2], date[1], date[0], time[0], time[1], time[2], time[3])

    def to_dict(self):
        return {
            "name": self.name,
            "tag": self.tag,
            "timestamp": self.timestamp,
            "date": None,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "activity": self.activity
        }

    def to_json(self):
        return json.dumps(self.to_dict())
