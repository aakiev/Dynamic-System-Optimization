from scipy.optimize import linprog

c = [-45,-53,-37,12,12,0,0,0,0,0,0]
A = [[1,0,0,0,0,-1,0,0,0,0,0],
     [0,1,0,0,0,0,-1,0,0,0,0],
     [0,0,1,0,0,0,0,-1,0,0,0],
     [8,10,5,-1,0,0,0,0,1,0,0],
     [4,7,6,0,-1,0,0,0,0,1,0], 
     [0,0,0,1,1,0,0,0,0,0,1]]
b = [75,75,75,3400,2700,1000]

# >= 0 Bedingungen müssen nicht seperat eingegeben werden!

res = linprog(c, A_eq=A, b_eq=b, method = "highs", integrality = [1,1,1,1,1,1,1,1,1,1,1])
print(res.message)
print(res.success)
print(res.fun)
print(res.x)