import random


def generate_service_type():
    weight = {"A": 0.3, "B": 0.3, "C": 0.4}
    return random.choices(list(weight.keys()), weights=list(weight.values()), k=1)[0]


def generate_difficulty():
    return random.randint(1, 3)


def generate_energy():
    return random.uniform(100, 200)


def generate_imitate_pro():
    return random.random()

def generate_cooperation():
    return random.randint(0,1)