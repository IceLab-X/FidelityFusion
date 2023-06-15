import torch
# TODO check minus eigen value

class eigen_pairs:
    def __init__(self, matrix) -> None:
        # eigen_value, eigen_vector = torch.symeig(matrix, eigenvectors=True)
        # if not torch.all(torch.linalg.eigvalsh(matrix)>0):
        #     print('WARNING: matrix may not be positive definite')
        eigen_value, eigen_vector = torch.linalg.eigh(matrix, UPLO='U')
        # UPLO='U'
        self.value = eigen_value
        self.vector = eigen_vector