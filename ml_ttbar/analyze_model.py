import os 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from CONSTANTS import *



def analyze_jets_1():
    model_dir = 'trained_models_low_pt/' 
    assert os.path.isdir(model_dir)

    # ---------------------------------

    history_logi, history_mlp, history_cnn = np.load(model_dir + 'training_histories.npz', allow_pickle=True)['arr_0']

    y_test = np.load(model_dir + 'y_test.npz')['arr_0']

    predictions0 = np.load(model_dir + 'predictions0.npz')['arr_0']
    predictions1 = np.load(model_dir + 'predictions1.npz')['arr_0']
    predictions_cnn = np.load(model_dir + 'predictions_cnn.npz')['arr_0']

    # ---------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplots_adjust(left=0.05, right=0.99, wspace=0.7) 
    labels=['Logistic', 'MLP', 'CNN']
    color = ['blue', 'green', 'red']
    nn = min([min([len(history[attr]) for attr in ['accuracy', 'val_accuracy', 'loss', 'val_loss']]) for history in [history_logi, history_mlp, history_cnn]])
 

    for i, history in enumerate([history_logi, history_mlp, history_cnn]):
        axes[1].plot(history['accuracy'][:nn], c=color[i], label=labels[i])
        axes[1].plot(history['val_accuracy'][:nn], c=color[i],  ls='--')
        axes[0].plot(history['loss'][:nn], c=color[i], label=labels[i])
        axes[0].plot(history['val_loss'][:nn], c=color[i], ls='--')
    
    axes[1].scatter(nn, 1. * sum(np.argmax(predictions0, axis=1) == np.argmax(y_test, axis=1)) / len(y_test), color='blue')
    axes[1].scatter(nn, 1. * sum(np.argmax(predictions1, axis=1) == np.argmax(y_test, axis=1)) / len(y_test), color='green')
    axes[1].scatter(nn, 1. * sum(np.argmax(predictions_cnn, axis=1) == np.argmax(y_test, axis=1)) / len(y_test), color='red')
    
    # Coordinates
    for iax in range(2):
        axes[iax].set(xlabel='Epoch', ylabel=['Loss', 'Accuracy'][iax])

    # Legend
    logi_line = plt.Line2D([], [], color='blue', linestyle='-', linewidth=1, label=labels[0])
    mlp_line = plt.Line2D([], [], color='green', linestyle='-', linewidth=1, label=labels[1])
    cnn_line = plt.Line2D([], [], color='red', linestyle='-', linewidth=1, label=labels[2])
    train_line = plt.Line2D([], [], color='black', linestyle=':', linewidth=1, label='train')
    validation_line = plt.Line2D([], [], color='black', linestyle='-', linewidth=1, label='validation')
    
    for i in range (2): 
        axes[i].legend(handles =[logi_line, mlp_line, cnn_line, train_line, validation_line], loc='center left', bbox_to_anchor=(1, 0.5))


    fig.savefig('training_history_all_epoch.png')

def analyze_jets_2():
    model_dir = 'trained_models_low_pt/' 
    assert os.path.isdir(model_dir)

    # ---------------------------------

    y_test = np.load(model_dir + 'y_test.npz')['arr_0']

    predictions0 = np.load(model_dir + 'predictions0.npz')['arr_0']
    predictions1 = np.load(model_dir + 'predictions1.npz')['arr_0']
    predictions_cnn = np.load(model_dir + 'predictions_cnn.npz')['arr_0']

    # ---------------------------------

    from sklearn.metrics import roc_curve
    fpr0, tpr0, _ = roc_curve(y_test.ravel(), predictions0.ravel())
    fpr1, tpr1, _ = roc_curve(y_test.ravel(), predictions1.ravel())
    fpr_cnn, tpr_cnn, _ = roc_curve(y_test.ravel(), predictions_cnn.ravel())

    from sklearn.metrics import auc
    auc0 = auc(fpr0, tpr0)
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr_cnn, tpr_cnn)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr0, tpr0, label='Logistic regression (area = {:.3f})'.format(auc0))
    plt.plot(fpr1, tpr1, label='Multilayer Perceptron (area = {:.3f})'.format(auc1))
    plt.plot(fpr_cnn, tpr_cnn, label='CNN (area = {:.3f})'.format(auc2))
    plt.gca().set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve', xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    plt.grid(True, which="both")
    plt.legend(loc='center left', bbox_to_anchor= (1, 0.5))
    plt.tight_layout()
    plt.savefig('ROC_curve.png')

def analyze_jets_3():
    model_dir = 'trained_models_low_pt/'
    assert os.path.isdir(model_dir)

    # ---------------------------------

    y_test = np.load(model_dir + 'y_test.npz')['arr_0']

    predictions0 = np.load(model_dir + 'predictions0.npz')['arr_0']
    predictions1 = np.load(model_dir + 'predictions1.npz')['arr_0']
    predictions_cnn = np.load(model_dir + 'predictions_cnn.npz')['arr_0']

    # ---------------------------------

    from sklearn.metrics import roc_curve
    fpr0, tpr0, thresholds = roc_curve(y_test.ravel(), predictions0.ravel())
    fpr1, tpr1, thresholds = roc_curve(y_test.ravel(), predictions1.ravel())
    fpr_cnn, tpr_cnn, thresholds = roc_curve(y_test.ravel(), predictions_cnn.ravel())

    from sklearn.metrics import auc
    auc0 = auc(fpr0, tpr0)
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr_cnn, tpr_cnn)

    np.seterr(divide='ignore', invalid='ignore')  # disable warning for 1/0 divisions
    plt.plot(thresholds, 1 / thresholds, 'k--')
    plt.plot(tpr0, 1 / fpr0, label='Logistic regression (area = {:.3f})'.format(auc0))
    plt.plot(tpr1, 1 / fpr1, label='Multilayer perceptron (area = {:.3f})'.format(auc1))
    plt.plot(tpr_cnn, 1 / fpr_cnn, label='CNN (area = {:.3f})'.format(auc2))
    plt.gca().set(ylabel='1/$\epsilon_B$ - 1/False positive rate', xlabel='$\epsilon_S$ - True positive rate', title='Tagging efficiency vs. background rejection', xlim=(-0.01, 1.01), ylim=(1, 5 * 10**3), yscale='log')
    plt.grid(True, which="both")
    plt.legend(loc='upper right')
    plt.savefig('ROC_curve_bg_rej.png')


def analyze_jets_4():
    model_dir = 'trained_models_low_pt/' 
    assert os.path.isdir(model_dir)

    # ---------------------------------

    x_test = np.load(model_dir + 'x_test.npz')['arr_0']
    y_test = np.load(model_dir + 'y_test.npz')['arr_0']

    predictions_cnn = np.load(model_dir + 'predictions_cnn.npz')['arr_0']

    # ---------------------------------

    logscale = True

    vmax = {} if logscale else dict(vmax=1)
    logscale = dict(norm=mpl.colors.LogNorm()) if logscale else {}

    fig, axes = plt.subplots(1, 5, figsize=(20, 2.5))
    for i in range(2):
        im = axes[i].imshow(x_test[predictions_cnn.argmin(axis=0)[i], :, :, 0], **vmax, cmap=cmap, **logscale)
        plt.colorbar(im, ax=axes[i])
        axes[i].set(title='most {}-like jet'.format(['top', 'qcd'][i]))
    axes[2].imshow(x_test[abs(predictions_cnn - 0.5).argmin(axis=0)[0], :, :, 0], **vmax, cmap=cmap, **logscale)
    plt.colorbar(im, ax=axes[2])
    axes[2].set(title='most uncertain jet')

    for iax, i in enumerate((predictions_cnn - y_test).argmin(axis=0)):
        im = axes[iax + 3].imshow(x_test[i, :, :, 0], **vmax, cmap=cmap, **logscale)
        plt.colorbar(im, ax=axes[iax + 3])
        axes[iax + 3].set_title('Output: {}\nLabel = {}    --FAIL--'.format(predictions_cnn[i], y_test[i]))

    fig.tight_layout()
    fig.savefig('cnn_jet_sample.png', dpi=150)


def analyze_empty_image():
    model_dir = 'trained_models_low_pt/' 
    assert os.path.isdir(model_dir)

    # ---------------------------------

    import keras

    model0 = keras.models.load_model(model_dir + 'logi.h5')
    model1 = keras.models.load_model(model_dir + 'mlp.h5')
    model_cnn = keras.models.load_model(model_dir + 'cnn.h5')

    x_test = np.load(model_dir + 'x_test.npz')['arr_0']

    # ---------------------------------

    empty = np.zeros(x_test[0].shape)
    print([model.predict(np.array([empty])) for model in [model0, model1, model_cnn]])

    
def check_image_brightness():
    model_dir = 'trained_models_low_pt/' 
    assert os.path.isdir(model_dir)

    # ---------------------------------

    x_test = np.load(model_dir + 'x_test.npz')['arr_0']
    y_test = np.load(model_dir + 'y_test.npz')['arr_0']

    # ---------------------------------

    plt.hist(list(map(np.sum, [x for x, y in zip(x_test, y_test) if np.argmax(y) == 0])), bins=np.arange(0, 100, 0.5), alpha=0.5, label='QCD')
    plt.hist(list(map(np.sum, [x for x, y in zip(x_test, y_test) if np.argmax(y) == 1])), bins=np.arange(0, 100, 0.5), alpha=0.5, label='top')
    plt.legend()

    plt.show()