from sklearn.metrics.pairwise import *


def kernelTrans(X, paras):
    m = len(X)
    K = np.zeros((m, m), dtype=np.float)
    paras[1] = float(paras[1])

    if paras[0]=='linear':
        K = linear_kernel(X, X)

    elif paras[0] == 'cov':
        K = np.cov(X)

    elif paras[0]=='rbf':
        K = rbf_kernel(X, X, paras[1] ** -2)
 
    elif paras[0] == 'laplace':
        K = laplacian_kernel(X, X, paras[1])
 
    elif paras[0] == 'poly':
        K = polynomial_kernel(X, X, paras[1])

    elif paras[0] == 'sigmoid':
        K = sigmoid_kernel(X, X, paras[1])

    elif paras[0] == 'chi2':
        K = chi2_kernel(X, X, paras[1])

    else: raise NameError('invalid kernel')

    return K