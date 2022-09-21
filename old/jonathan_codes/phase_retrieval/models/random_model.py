import numpy as np


def random_sampling(n=1024, d=256, k=1, x_scale=1, A_scale=1, Delta=0, is_complex=0, verbose=0):
	"""
	Generates data y = |A x|^2 with A and x random complex gaussian matrices / vectors
	n is the number of measurements
	d is the dimension of x
	k is the number of phase retrieval problems that we solve in parallel
	Delta is the variance of an additive gaussian noise
	"""
	if verbose:
		print("Generating random data following the model y = |Ax|^2... ")
	if is_complex == 0:
		x = x_scale * np.random.randn(d, k)  # x_scale is the std of each component of x
		A = A_scale * np.random.randn(n, d) / np.sqrt(d)  # A_scale is the expected norm of every row of A
	else:
		x = x_scale * (np.random.randn(d, k) + 1j * np.random.randn(d, k)) / np.sqrt(2)
		# x_scale is the std of each component of x
		A = A_scale * (np.random.randn(n, d) + 1j * np.random.randn(n, d)) / np.sqrt(2 * d)
		# A_scale is the expected norm of every row of A
	y = np.abs(A @ x)**2 + np.sqrt(Delta) * np.random.randn(n, k)
	# y follows an exponential distribution of mean (x_scale * A_scale)^2
	# proof steps: z := sum a_j x_j, Var(a_j x_j) = 1/d, Var(z) = 1 = E(z^2), y = z^2, homogeneity argument

	if verbose:
		print("Data generation complete")
	return y, A, x

