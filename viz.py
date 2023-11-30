from utils import *


path = './sa1b'
dataset = SA1B_Dataset(path, transform=input_transforms, target_transform=target_transforms)
image, target = dataset.__getitem__(3535)


image = input_reverse_transforms(image)
image = np.array(image)

selected_target = target[8]

plt.imshow(image)
plt.imshow(target[8], alpha=0.5)

random_coordinate = point_sample(None, selected_target)
plt.scatter(random_coordinate[1], random_coordinate[0], color='b', s=50)

plt.show()