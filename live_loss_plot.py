import matplotlib.pyplot as plt
from keras.callbacks import Callback
from IPython.display import clear_output
#from matplotlib.ticker import FormatStrFormatter

# TODO
# object-oriented API


def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Log-loss (cost function)"}
    if x in translations:
        return translations[x]
    else:
        return x
def plots_slot1(history):
    # summarize history for accuracy

    plt.plot(history.history['dense_4_f1'])
    plt.plot(history.history['val_dense_4_f1'])
    plt.title('model f1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['dense_4_loss'])
    plt.plot(history.history['val_dense_4_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plots_slot1_t3(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('model f1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.savefig('f1.png')
    plt.show()


def entity_attribute_detection_plots(history):
    # E#A pair loss and f1
    plt.plot(history.history['ea_pair_loss'])
    plt.plot(history.history['val_ea_pair_loss'])
    plt.title('Model E#A loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('ea_loss.png')
    plt.show()
    plt.plot(history.history['ea_pair_f1'])
    plt.plot(history.history['val_ea_pair_f1'])
    plt.title('Model E#A f1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('ea_f1.png')
    plt.show()

    # entities loss and f1
    plt.plot(history.history['entities_loss'])
    plt.plot(history.history['val_entities_loss'])
    plt.title('Model entities loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('entities_loss.png')
    plt.show()
    plt.plot(history.history['entities_f1'])
    plt.plot(history.history['val_entities_f1'])
    plt.title('Model entities f1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('entities_f1.png')
    plt.show()

    # attributes loss and f1
    plt.plot(history.history['attributes_loss'])
    plt.plot(history.history['val_attributes_loss'])
    plt.title('Model attributes loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('attributes_loss.png')
    plt.show()
    plt.plot(history.history['attributes_f1'])
    plt.plot(history.history['val_attributes_f1'])
    plt.title('Model attributes f1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('attributes_f1.png')
    plt.show()


def plots(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()




class PlotLosses(Callback):
    def __init__(self, figsize=None):
        super(PlotLosses, self).__init__()
        self.figsize = figsize

    def on_train_begin(self, logs={}):

        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)

        clear_output(wait=False)
        plt.figure(figsize=self.figsize)
        
        for metric_id, metric in enumerate(self.base_metrics):
            plt.subplot(1, len(self.base_metrics), metric_id + 1)
            
            plt.plot(range(1, len(self.logs) + 1),
                     [log[metric] for log in self.logs],
                     lw=2, label="training")
            if self.params['do_validation']:
                plt.plot(range(1, len(self.logs) + 1),
                         [log['val_' + metric] for log in self.logs],
                         lw=2, label="validation")
            plt.title(translate_metric(metric))
            plt.xlabel('epoch')
            plt.legend(loc='center right')
        
        plt.tight_layout()
        plt.show()

    def on_train_end(self, logs=None):
        #plt.show()
         print("finished training")

