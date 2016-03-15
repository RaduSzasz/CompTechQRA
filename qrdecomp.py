import numpy as np
import sys
import getopt

debug = open ('debug.txt', 'w')

#######################################################################
####### Reads the file provided via a command line argument and #######
####### returns a np.matrix ###########################################
#######################################################################
def read_file (file_name):
    file = open (file_name, 'r')

    rows = file.readlines ()
    matrix = ';'.join (rows)

    # Making sure the values are cast to double precision so that
    # we can avoid precision problems further on
    return np.array (np.mat(matrix), dtype = np.float64)

def generate_rand_sym_matrix (N):
    matrix = np.empty (shape = (N, N), dtype = np.float64)
    for i in range (N):
        for j in range (i, N):
            matrix[i][j] = matrix[j][i] = np.random.rand ()
    return matrix

#######################################################################
####### Generates symmetric matrix either from file name provided #####
####### or randomly based on the dimension given. Includes checking ###
#######################################################################
def generate_matrix (file_name, dimension):
    matrix = ''
    if file_name:
        matrix = read_file (file_name)
    else:
        matrix = generate_rand_sym_matrix (dimension)

    if not np.all(matrix.T ==  matrix):
        print ('Matrix should be symmetric')
        print ('Something went wrong :(')
        sys.exit ()

    return matrix

#######################################################################
####### Displays message when argument parsing resulted in error ######
#######################################################################
def parse_error ():
    print ('qrdecomp -f <file_name> -d <dimension>')
    print ('In case both a file name and a dimesnion are present, only file name is considered')
    sys.exit ()


#######################################################################
####### Parses the command line arguments given to the program ########
####### and returns a pair consisting of file_name and desired ########
####### matrix dimension ##############################################
#######################################################################
def parse_arguments (argv):
    # Help, file, dimension
    options = 'hf:d:o:'
    try:
        opts, args = getopt.getopt (argv, options)
    except getopt.GetoptError:
        parse_error ()

    if not opts:
        parse_error ()

    file, output, dim = 0, 0, 0

    for opt, arg in opts:
        if opt == '-h':
            parse_error ()
        elif opt == '-f':
            file = arg
        elif opt == '-d':
            dim = int (arg)
        elif opt == '-o':
            output = arg
    return file, output, dim

def arr_str (arr):
    return np.array_str (arr, precision = 5, suppress_small=True)

def qr_decomposition (A):
    N = A.shape[0]
    Q = np.empty (shape = (N, N), dtype = np.float64)
    R = np.zeros (shape = (N, N), dtype = np.float64)

    debug.write ('We try to do QR decomposition for matrix:\n{0}\n'.format(arr_str (A)))

    for j in range (N):
        v = A[:,j]

        for i in range (j):
            R.itemset((i, j), np.dot (Q[:,i], v))
            v = np.subtract (v, R[i][j] * Q[:,i])

        v_norm = np.linalg.norm (v)
        R.itemset((j, j), v_norm)
        Q[:,j] = v / R[j][j]

    debug.write ('We get Q = \n{0}\n'.format (arr_str (Q)))
    debug.write ('We get R = \n{0}\n'.format (arr_str (R)))
    return Q, R

def get_next_qr_iteration (A):
    Q, R = qr_decomposition (A)
    nextA = np.dot (R, Q)

    return nextA


def get_eigenvalues_eigenvectors (A):
    MAX_ITERATIONS = 10
    overall_Q = np.identity (A.shape[0])
    for _ in range (MAX_ITERATIONS):
        Q, R = qr_decomposition (A)
        nextA = np.dot (R, Q)
        A = nextA
        overall_Q = np.dot (overall_Q, Q)

    eigenvalues = np.diag (A)
    eigenvectors = [overall_Q[:,i] for i in range (overall_Q.shape[1])]
    return eigenvalues, eigenvectors

def print_result (output, eigenvalues, eigenvectors):
    f = open (output, 'w')
    for (eigenvalue, eigenvector) in zip (eigenvalues, eigenvectors):
        f.write ('For eigenvector {0}\n\t we have eigenvalue {1}\n'.format(eigenvalue, np.array_str (eigenvector, precision = 4, suppress_small=True)))

def main (argv):
    file, output, dimension = parse_arguments (argv)
    A = generate_matrix (file, dimension)
    eigenvalues, eigenvectors = get_eigenvalues_eigenvectors (A)
    print_result (output, eigenvalues, eigenvectors)

if __name__ == "__main__":
    main (sys.argv[1:])
