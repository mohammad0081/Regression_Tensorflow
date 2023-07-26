import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(m_now, b_now, data : list, L):
    m_gradient = 0.0
    b_gradient = 0.0
    n = len(data)

    for i in range(n):
        x = data[i][0]
        y = data[i][1]

        m_gradient += (1/n) * x * (( m_now*x + b_now ) - y)
        b_gradient += (1/n) * (( m_now*x + b_now ) - y)

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient

    return m, b

n = 200
data = [(i , 2*i + 12 + 10*np.random.randint(-10, 10) ) for i in range(0, n)]

m, b = 0, 0

# first i tried L = 0.001 but after some iteration it returned None
# the i changed it to lower values like 0.00001 and it fixed
L = 0.00001
epochs = 500

m_before = 0
for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
    m_after = m
    if m_after - m_before < 0.001:
        break

    m_before = m
    if i%2 == 0 :
        print(f"Epoch : {i} m={m}, b={b}")

plt.scatter( [ item[0] for item in data], [ item[1] for item in data], color = 'black')
plt.plot(list(range(0, n)), [ m*x +b for x in range(0, n)], color = 'red')
plt.show()