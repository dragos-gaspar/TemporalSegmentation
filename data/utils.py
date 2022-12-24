from matplotlib import pyplot as plt


def is_overlap(a, b):
    return bool(max(0, min(a[1], b[1]) - max(a[0], b[0])))


def draw_pair(frame1, frame2):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(frame1)
    axs[1].imshow(frame2)
    plt.show()
