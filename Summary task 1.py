from timeit import repeat


X = [15,46.45,60.67,54,73,17,68,59,12,7,31,69,92]
y = [34.7,105.7,80.6,118.1,136.8,101.5,146.0,55.0,119.2,114.0,48.7,54.4,69.2,136.3,185.9]
J = []


def  calcJ(X,Y, a, b): 
    j,da,db = 0
    
    m = len(Y)
    
    for i in range(m):
        j =+ pow((a*X[i]+b -Y[i]), 2)
        
    j = j/m
    
    for i in range(m):
        db =+ 2(a * X[i]+b -y[i])
        da =+ 2*X[i]*(a * X[i]+b -y[i])
    
    da = da/ m
    db = db/ m

    return j, da, db 

def train (X,Y, alph, repeat):
    a,b = 3
    
    for i in range(repeat):
        j, da, db = calcJ(X,Y,a,b)
        J.append(j)
        a = a - da * alpha
        b = b - db * alpha
    print("a = ",a)
    print("b = ",b)
        
print(J)

train(X,Y, 0.0001,100)

