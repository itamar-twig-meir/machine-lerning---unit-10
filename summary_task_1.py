
#This is quite the shitty code. its old and im incredibly ashamed of it(but also to lazy to rewrite )

X = [15, 46, 45, 60, 67, 54, 73, 17, 68, 59, 12, 7, 31, 69, 92]
Y = [34.7, 105.7, 80.6, 118.1, 136.8, 101.5, 146.0, 55.0, 119.2, 114.0, 48.7, 54.4, 69.2, 136.3, 185.9]


def calcj(X, Y, a, b):
    j = 0
    da = 0
    db = 0

    m = len(Y)

    for i in range(m):
        j = j + (a * X[i] + b - Y[i]) ** 2
        db = db + 2 * (a * X[i] + b - Y[i])
        da = da + 2 * X[i] * (a * X[i] + b - Y[i])

    j = j / m
    da = da / m
    db = db / m

    return j, da, db


def train(X, Y, alph, repeat):
    a = 3
    b = 3

    for i in range(repeat):
        j, da, db = calcj(X, Y, a, b)
        a = a - da * alph
        b = b - db * alph
        if i % 100000 == 0:
            print(j)

    return a, b


a, b = train(X, Y, 0.00002, 2000000)
print("final a = "+ a)
print("final a = "+ b)
