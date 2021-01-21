import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology



def check(B, y, x):
    if not 0 <= x < B.shape[1]:
        return False
    if not 0 <= y < B.shape[0]:
        return False
    if B[y,x] != 0:
        return True
    return False

def neighbours2(B, y, x):
    left = y, x-1
    top = y-1, x
    if not check(B, *left):
        left = None
    if not check(B, *top):
        top = None    
    return left, top
  
def two_pass_labeling(B):
    links = np.zeros(B.size, dtype="uint")
    labeled = np.zeros_like(B)
    label = 1
    for row in range(B.shape[0]):
        for col in range(B.shape[1]):
            if B[row,col] != 0:
                ns = neighbours2(B, row, col)
                if ns[0] is None and ns[1] is None:
                    m = label
                    label += 1
                else:
                    labels  = [labeled[n] for n in ns if n is not None]
                    m = min(labels)
                labeled[row, col] = m
                for n in ns:
                    if n is not None:
                        lbl = labeled[n]
                        if lbl != m:
                            union(m,lbl,links)
    for row in range(B.shape[0]):
        for col in range(B.shape[1]):
            if B[row, col] != 0:
                new_label = find(labeled[row,col], links)
                if new_label != labeled[row,col]:
                    labeled[row,col] = new_label
    count = 0
    for i in np.unique(labeled):
        labeled[labeled == i] = count
        count +=1
    return labeled

def find(label, links):
    j = label
    while links[j] != 0:
        j = links[j]
    return j

def union(label1, label2, links):
    j = find(label1, links)
    k = find(label2, links)
    if j != k:
        links[k] = j
        
def array_to_ascii(array):
    to_ascii = array.copy()[::2,::2].astype('U')
    to_ascii[to_ascii == "1"] = "██"
    to_ascii[to_ascii == "0"] = "  "
    return "\n".join(("".join(r) for r in to_ascii))









struct1 = np.array([[1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1]])

image_orig = np.load("ps.npy.txt").astype("uint16")
image = image_orig.copy()
res = 0
print("\nКоличество объектов:\n")
for i in range(5):
    result = morphology.binary_opening(image, struct1).astype("uint16")
    num = np.max(two_pass_labeling(result))
    res += num
    print(f"{array_to_ascii(struct1)}: {num}\n")
    if not i:
        image = np.bitwise_xor(image, result)
        struct1[:2,2:4] = 0
    struct1 = np.rot90(struct1)
print(f"Общее количество объектов: {res}")


