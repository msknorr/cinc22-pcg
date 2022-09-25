import numpy as np

def get_most_audible(data):
    for i, l in enumerate(data.split('\n')):
        if l.split(" ")[0] == "#Most":
            return l.split(" ")[-1]


def get_hearable_leads(data):
    for i, l in enumerate(data.split('\n')):
        if l.split(" ")[0] == "#Murmur" and l.split(" ")[1] == "locations:":
            return l.split(" ")[-1].split("+")


def get_murmur_timing(data):  # 5
    # nan               637
    # Holosystolic       86
    # Early-systolic     45
    # Mid-systolic       16
    # Late-systolic       1
    to_one_hot = ["nan", "Holosystolic", "Early-systolic", "Mid-systolic", "Late-systolic"]
    for i, l in enumerate(data.split('\n')):
        if l.split(" ")[0] == "#Systolic" and l.split(" ")[1] == "murmur" and l.split(" ")[2] == "timing:":
            template = np.zeros(len(to_one_hot))
            template[to_one_hot.index(l.split(" ")[-1])] = 1
            return template, l.split(" ")[-1]


def get_murmur_shape(data):  # 5
    # nan            637
    # Plateau         93
    # Diamond         28
    # Decrescendo     25
    # Crescendo        2
    to_one_hot = ["nan", "Plateau", "Diamond", "Decrescendo", "Crescendo"]
    for i, l in enumerate(data.split('\n')):
        if l.split(" ")[0] == "#Systolic" and l.split(" ")[1] == "murmur" and l.split(" ")[2] == "shape:":
            template = np.zeros(len(to_one_hot))
            template[to_one_hot.index(l.split(" ")[-1])] = 1
            return template, l.split(" ")[-1]


def get_murmur_grading(data):  # 4
    # nan       637
    # I/VI       82
    # III/VI     40
    # II/VI      26
    to_one_hot = ["nan", "I/VI", "III/VI", "II/VI"]
    for i, l in enumerate(data.split('\n')):
        if l.split(" ")[0] == "#Systolic" and l.split(" ")[1] == "murmur" and l.split(" ")[2] == "grading:":
            template = np.zeros(len(to_one_hot))
            template[to_one_hot.index(l.split(" ")[-1])] = 1
            return template, l.split(" ")[-1]


def get_murmur_pitch(data):  # 4
    # nan       637
    # Low        68
    # Medium     44
    # High       36
    to_one_hot = ["nan", "Low", "Medium", "High"]
    for i, l in enumerate(data.split('\n')):
        if l.split(" ")[0] == "#Systolic" and l.split(" ")[1] == "murmur" and l.split(" ")[2] == "pitch:":
            template = np.zeros(len(to_one_hot))
            template[to_one_hot.index(l.split(" ")[-1])] = 1
            return template, l.split(" ")[-1]


def get_murmur_quality(data):  # 4
    # nan        637
    # Harsh       81
    # Blowing     64
    # Musical      3
    to_one_hot = ["nan", "Harsh", "Blowing", "Musical"]
    for i, l in enumerate(data.split('\n')):
        if l.split(" ")[0] == "#Systolic" and l.split(" ")[1] == "murmur" and l.split(" ")[2] == "quality:":
            template = np.zeros(len(to_one_hot))
            template[to_one_hot.index(l.split(" ")[-1])] = 1
            return template, l.split(" ")[-1]