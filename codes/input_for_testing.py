import numpy as np

local_path = "../codes"

ldpc_girth_min_6 = {
    10: np.loadtxt(local_path + "/mkmn_codes/mkmn_24_6_10.txt", delimiter=" ").astype(
        int
    ),
    8: np.loadtxt(local_path + "/mkmn_codes/mkmn_20_5_8.txt", delimiter=" ").astype(
        int
    ),
    6: np.loadtxt(local_path + "/mkmn_codes/mkmn_16_4_6.txt", delimiter=" ").astype(
        int
    ),
}

ldpc_56 = {
    10: np.loadtxt(local_path + "/5-6/24_4_10.txt", delimiter=" ").astype(int),
    8: np.loadtxt(local_path + "/5-6/18_4_8.txt", delimiter=" ").astype(int),
    6: np.loadtxt(local_path + "/5-6/12_2_6.txt", delimiter=" ").astype(int),
}

hamming = np.asarray(
    [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
).T
hamming_deg = np.vstack([hamming, [1, 0, 1, 0, 1, 0, 1]])
hamming_mds = np.asarray(
    [
        [1, 1, 0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
)
self_dual = np.asarray(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 1, 0],
    ]
).T


if __name__ == "__main__":
    print(hamming_deg)
    print(ldpc_girth_min_6.get(6))
