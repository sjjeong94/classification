import cv2
import datasets
import transforms


def test():
    T_compose = transforms.Compose([
        transforms.Pad(2),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
    ])
    dataset = datasets.CIFAR10('./data', True, T_compose)

    idx = 0
    while True:
        image, label = dataset[idx]
        print(image.shape)
        cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(datasets.classes[label])
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('q'):
            idx -= 1
        else:
            idx += 1

        if idx < 0:
            idx = len(dataset)-1
        elif idx >= len(dataset):
            idx = 0

    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
