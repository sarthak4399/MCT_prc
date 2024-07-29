import numpy as np
import matplotlib.pyplot as plt
# 1st program
a = np.array([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]])
row, col = a.shape

# 2nd program
print(a[1, 2]) # 2nd row 3rd col
print(a[:, 1]) # 2nd col complete
print(a[2, :]) # 3rd row complete
print(a[:, 1:3]) # 2nd & 3rd col
print(a[1:4, 1:3]) # 2nd to 4th row & 2nd and 3rd col
print(a[:, 0:3:2]) # 1st and 3rd col with diff of 2
print(a.flat[11]) # 12th element counting vertically
print(a.flatten()) # all elements in vertical line
print(a) # as it is
print(np.diag(a)) # diagonal elements
print(np.diag(a, 1)) # diag shifted upward by 1
print(np.diag(a, -1)) # diag shifted downwards
print(np.diag(a, 2)) # diag shifted upwards by 2

# 3rd program
b1 = np.zeros((2, 3))
b2 = np.ones((2, 2))
b3 = np.eye(3) # identity
b4 = np.random.rand(1, 5) # uniform distribution
b5 = np.random.randn(5, 5) # normal distribution

# 4th program: Matrix Concatenation
B = np.arange(1, 7)
C = np.arange(1, 5)
C = np.concatenate((B, C))
B = np.concatenate((a, np.vstack((np.ones((2, 2)), np.eye(2)))), axis=1)
C2 = np.array([[12], [34]])
D = np.tile(C2, (2, 3))

# 5th program
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 6]])
print(A ** 2)
print(np.linalg.matrix_power(A, 2))
print(A.T)
print(np.linalg.inv(A))
print(np.linalg.matrix_rank(A))
print(np.poly(A)) # Coefficients of characteristic polynomial
print(np.roots(np.poly(A))) # eigen values or roots of above poly

# (iii) part
# 1st program: plot sine
time = np.arange(0, 2 * np.pi, 0.1)
w1 = np.sin(time)
plt.plot(time, w1)
plt.grid(True)
plt.show()

# 2nd program: plot cosine in same figure
w2 = np.cos(time)
plt.plot(time, w1, label='sin(t)')
plt.plot(time, w2, label='cos(t)')
plt.grid(True)
plt.title('sine & cosine plot')
plt.xlabel('time')
plt.ylabel('sin(t) & cos(t)')
plt.legend()
plt.show()

# 3rd program: subplot & Lissajous patterns
plt.figure()

plt.subplot(2, 2, 1)
plt.plot(np.sin(time), np.cos(time))
plt.title('Lissajous Pattern')
plt.xlabel('cos(t)')
plt.ylabel('sin(t)')

plt.subplot(2, 2, 2)
plt.plot(np.sin(2 * time), np.cos(time))
plt.title('Lissajous Pattern')
plt.xlabel('cos(t)')
plt.ylabel('sin(2t)')

plt.subplot(2, 2, 3)
plt.plot(np.sin(3 * time), np.cos(time))
plt.title('Lissajous Pattern')
plt.xlabel('cos(t)')
plt.ylabel('sin(3t)')

plt.subplot(2, 2, 4)
plt.plot(np.sin(4 * time), np.cos(time))
plt.title('Lissajous Pattern')
plt.xlabel('cos(t)')
plt.ylabel('sin(4t)')

plt.tight_layout()
plt.show()

# 4th program: 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(np.sin(3 * time), np.cos(3 * time), time)
ax.grid(True)
ax.set_title('3D plot of Helix')
ax.set_xlabel('sin(t)')
ax.set_ylabel('cos(t)')
ax.set_zlabel('time')
plt.show()
