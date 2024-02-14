
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

class GeneratePDF:
  def __init__(self): pass

  def create(self, unet, pdf_path, history):
    with PdfPages(pdf_path) as pdf:
      firstPage = plt.figure(figsize=(unet.args.fig_sizex, unet.args.fig_sizey))
      text = "Model Analysis"
      firstPage.text(0.5, 0.5, text, size=24, ha="center")
      pdf.savefig()

      #summarize history for binary accuracy
      plt.figure(figsize = (unet.args.fig_sizex, unet.args.fig_sizey))
      plt.subplot(unet.args.subplotx, unet.args.subploty, 1)
      plt.plot(history.history['accuracy'])
      plt.plot(history.history['val_accuracy'])
      plt.title('Binary Accuracy')
      plt.legend(['train', 'val'], loc='upper left')

      # summarize history for loss
      plt.subplot(unet.args.subplotx, unet.args.subploty, 2)
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('Loss')
      plt.legend(['train', 'val'], loc='upper left')
    #  pdf.savefig()

      #summarize dice coefficient
      plt.subplot(unet.args.subplotx, unet.args.subploty, 3)
      plt.plot(history.history['iou_score'])
      plt.plot(history.history['val_iou_score'])
      plt.title('IOU Score')
      plt.legend(['train', 'val'], loc='upper left')
      pdf.savefig()