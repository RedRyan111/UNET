from matplotlib import pyplot as plt


def display_image(iteration, display_every, loss_list, rgb_predicted, target_features, target_img):
    iternums = [i * display_every for i in range(len(loss_list))]
    plt.figure(figsize=(20, 4))
    plt.subplot(141)
    plt.imshow(target_img.detach().cpu().numpy()[0])
    plt.title(f"Target Image")
    plt.subplot(142)
    plt.imshow(target_features.detach().cpu().numpy()[0])
    plt.title(f"Target Features")
    plt.subplot(143)
    plt.imshow(rgb_predicted.detach().cpu().numpy()[0])
    plt.title(f"Prediction at iteration {iteration}")
    plt.subplot(144)
    plt.plot(iternums, loss_list)
    plt.title("Loss")
    plt.show()