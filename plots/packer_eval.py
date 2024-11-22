import re
import numpy as np
from matplotlib import pyplot as plt


def get_data(path):
    # Sample input data (as if read from a text file)
    with open(path, "r") as f:
        data = f.readlines()

    data = "\n".join(data)

    # Extracting areaR and utilisation values from the data
    areaR_list = []
    utilisation_list = []

    # Regular expression to match 'areaR' and 'utilisation'
    pattern = r"areaR=([\d.]+)\s+utilisation=([\d.]+)"

    matches = re.findall(pattern, data)

    for match in matches:
        areaR_list.append(float(match[0]))
        utilisation_list.append(float(match[1]))

    # Convert to numpy arrays for easy statistical calculations
    areaR_array = np.array(areaR_list)
    utilisation_array = np.array(utilisation_list)

    return areaR_array, utilisation_array


r1, u1 = get_data("../data/packer/d1.txt")
r2, u2 = get_data("../data/packer/d2.txt")
r3, u3 = get_data("../data/packer/d3.txt")
r11, u11 = get_data("../data/packer/d11.txt")

print(np.mean(u1), np.mean(u2), np.mean(u3), np.mean(u11))
print(np.mean(np.concatenate([u1, u2, u3, u11])))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.boxplot([u1,u2,u3,u11])
ax.set_xticklabels(['FIELD1', 'FIELD2', 'FIELD3', 'FIELD4'])
ax.set_xlabel('Field')
ax.set_ylabel('Utilisation Rate')

# show plot
plt.show()
