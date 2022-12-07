import matplotlib.pyplot as plt
import numpy as np

def imagePlot(image, boxes = None ):
    assert len(image.shape) == 3
    plt.imshow(np.transpose(image, (1,2,0)))
    if boxes:
        ax = plt.gca()
        ax.add_patch(
            plt.Rectangle(xy=(boxes[0], boxes[1]), width=boxes[2]-boxes[0], height=boxes[3]-boxes[1],
                          fill=False, edgecolor='blue', linewidth=2))