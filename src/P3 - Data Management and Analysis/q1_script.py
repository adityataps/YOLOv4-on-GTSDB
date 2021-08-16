#####################################
# q1_script.py                      #
# Aditya Tapshalkar                 #
# Georgia Institute of Technology   #
# Summer/Fall 2021                  #
#####################################


import json
import matplotlib as plt
plt.rcParams['figure.dpi'] = 500
import matplotlib.image as mplimg
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches

# Parsing json to get "pn" predicted images
def parse_json():
    pn_images = []
    with open("GTSDB.json", "r") as json_file:
        data = json.load(json_file)
        num_frames = len(data["output"]["frames"])
        for frame in range(num_frames):
            for sign in data["output"]["frames"][frame]["signs"]:
                if sign["class"] == "pn":
                    pn_images.append((data["output"]["frames"][frame]["frame_number"],
                                      sign["coordinates"],
                                      sign["detection_confidence"]))
    return pn_images

# Appending ground truths file to include "pn" class
def inject_gt(pn_images):

    with open("gt.txt", "a") as gt_txt:
        for image in pn_images:
            print("___________")
            show_img(image)
            input_class = input("Which class? (43 for ❌, 44 for 🚫, -1 for skip): ")
            if input_class == "-1":
                continue
            else:
                inject_data = f"{image[0]};{image[1][0]};{image[1][1]};{image[1][0] + image[1][2]};{image[1][1] + image[1][3]};{input_class}\n"
                print("\n", inject_data)
                gt_txt.write(inject_data)

# Showing the image with Matplotlib for annotating
def show_img(img):
    img_name, coords, confidence = img
    img = mplimg.imread("../../data/FullIJCNN2013/FullIJCNN2013/" + img_name)
    pyplot.imshow(img)
    rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=1, edgecolor="r", facecolor="none")
    pyplot.gca().add_patch(rect)
    pyplot.show()
    print(f"Image name: \t{img_name}")
    print(f"Coordinates: \t{coords}")
    print(f"Confidence: \t{confidence}")


if __name__ == '__main__':
    pn_images = parse_json()
    inject_gt(pn_images)
