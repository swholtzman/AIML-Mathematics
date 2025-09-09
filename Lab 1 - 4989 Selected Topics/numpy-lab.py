import numpy as np

# numpy.array
# Creates a numpy array from a list or sequence
# example
print("---+--- Create numpy array ---+---")
a = np.array([1, 2, 3])
print("↓ a = np.array([1, 2, 3]) ↓ \n", a)


b = np.array([[1, 2, 3],[4, 5, 6],[1, 4, 6]])
print("↓ b = np.array([[1, 2, 3],[4, 5, 6],[1, 4, 6]]) ↓ \n", b)
print("---+--- ---+--- ---+--- ---+--- ---+--- \n")

# numpy.equal
# compares two arrays element-wise, returns a boolean array (True if equal)
# example
print("---+--- Check array equality ---+---")
a = [1,2,3,4,5]
numa = np.array(a)
b = [6,7,8,9,10]
numb = np.array(b)

print("a = ", a)
print("numa = np.array(a) =", numa, "\n")

print("b = ", b)
print("numa = np.array(b) =", numb, "\n")

print("↓ np.equal(numa,numb).all() ↓ \n", np.equal(numa,numb).all())
print("---+--- ---+--- ---+--- ---+--- ---+--- \n")

# numpy.eye
# creates a 2D identity matrix with 1s on the diagonal
# example; create a 4x4 identity matrix:
print("---+--- Create a 4x4 identity matrix ---+---")
b = np.eye(4)
print("↓ b = np.eye(4) ↓ \n", b)
print("---+--- ---+--- ---+--- ---+--- ---+--- \n")

# numpy.dot
# computes the dot product (matrix multiplication or vector inner product)


# numpy.transpose
# swaps the rows and columns of an array (flips dimensions)
# example
print("---+--- Transpose the Matrix ---+---")
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("↓ a = np.array([[1,2,3],[4,5,6],[7,8,9]]) ↓ \n", a)

aTrans = a.transpose()
print("↓ a.transpose() ↓ \n", aTrans)

print("---+--- ---+--- ---+--- ---+--- ---+--- \n")

# numpy.linalg.norm
# calculates the length (magnitude) of a vector or matrix norm

# numpy.linalg.det
# returns the determinant of a square matrix

# numpy.linalg.inv
# returns the inverse of a square matrix (if it exists)
# example
print("---+--- Return the inverse matrix ---+---")
a = np.array([[0, 5],[1, 8]])
print("↓ a = np.array([[0, 5],[1, 8]]) ↓\n", a)
a_inv = np.linalg.inv(a)
print("↓ a_inv = np.linalg.inv(a) ↓\n", a_inv)

print("\n")

b = np.array([[0.6, 0.3],[-1.6, -0.8]])
print("↓ b = np.array([[0.6, 0.3],[-1.6, -0.8]]) ↓\n", b)
b_inv = np.linalg.inv(b)
print("↓ b_inv = np.linalg.inv(b) ↓\n", b_inv)
print("---+--- ---+--- ---+--- ---+--- ---+--- \n")

# numpy.linalg.solve
# solves a system of linear equations (Ax = b) for x

# numpy.allclose()
# compares values and returns if they are equal within a certain threshold
# example
print("---+--- All Close () ---+---")
x = np.array([1.0, 2.0, 3.0000001])
print(x)
y = np.array([1.0, 2.0, 3.0])
print(y)
print("np.allclose(x, y) = ", np.allclose(x, y))
print("---+--- ---+--- ---+--- ---+--- ---+--- \n")

# x.shape[i]

# Orthogonal Matrix Check
print("---+--- Check for an Orthogonal Matrix ---+---")
print("---+--- AA^t = I = A^tA ---+---")

a = np.array([[3,1,0],[1,1,4],[0,4,0]])
print("A = \n", a, "\n____")

a_trans = a.transpose()
print("A^t = \n", a_trans, "\n____")

resultant_a = np.dot(a, a_trans)
print("↓ A * A^t = ↓\n", resultant_a)

print("--- Is this orthogonal? ---")
identity = np.eye(a.shape[0])
print("↓ identity = np.eye(a.shape[0]) = ↓\n", identity)
is_ortho = np.allclose(resultant_a, identity)
print("is_ortho = np.allclose(resultant_a, identity) = \n", is_ortho)
print("---+--- ---+--- ---+--- ---+--- ---+--- \n")

# Symmetrical Matrix
# Is the Matrix identical to its transpose?