from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showerror, showinfo
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
import os.path
from os import path
import numpy as np

import file_inout as fio

"""   TODO: 
      - implement extraction for the ExtractBeads list of selected beads in ExtractBeads
      - implement beads intensity centering in Centering beads( possible use of existing method from pevious image manipulation)
      - implement averaging by gaussian blur.
"""



class BeadExtraction(Tk):
      """Class provides instruments for extraction of beads from microscope multilayer photo."""

      def __init__(self, master = None, wwidth=600, wheight = 600):
            super().__init__()
            # new  class properties
            self.beadCoords = [] # Coordinates of beads on the canvas
            self.beadMarks = []
            self.sideHalf = 18
            self.xr = 0
            self.yr = 0
            self.beadImPath = ''

            # new window widgets
            self.title("Bead extraction window.")
            self.resizable(False,False)

            self.cnv1 = Canvas(self,  width = wwidth, height = wheight, bg = 'white')
            self.cnv1.grid(row = 0,column=0, columnspan=2,sticky=(N,E,S,W))
            self.hScroll = Scrollbar(self, orient = 'horizontal')
            self.vScroll = Scrollbar(self, orient = 'vertical') 
            self.hScroll.grid(row = 1,column=0,columnspan=2,sticky=(E,W))
            self.vScroll.grid(row=0,column=2,sticky=(N,S))
            self.hScroll.config(command = self.cnv1.xview)
            self.cnv1.config(xscrollcommand=self.hScroll.set)
            self.vScroll.config(command = self.cnv1.yview)
            self.cnv1.config(yscrollcommand=self.vScroll.set)

            Button(self, text = 'Load Beads Photo', command = self.LoadBeadsPhoto).grid(row=2,column=0,sticky='we')

            Button(self,text = 'Undo mark', command = self.RemoveLastMark).grid(row =2,column = 1,sticky='we')
            
            Label(self, text = 'Selection Size: ').grid(row=3,column=0,sticky='e')
            self.selectSizeEntry = Entry(self, width = 5, bg = 'green', fg = 'white')

            self.selectSizeEntry.grid(row = 3, column = 1, sticky = 'w')
            self.selectSizeEntry.insert(0,self.sideHalf * 2)
            self.selectSizeEntry.bind('<Return>', self.ReturnSizeEntryContent)
            self.cnv1.bind('<Button-3>', self.BeadMarkClick)

            Button(self, text = 'Print marks coordinate list', command = self.PrintBeadsList).grid(row=4,column=0,sticky='we')

            Button(self, text = 'Clear All Marks', command = self.ClearAllMarks).grid(row=5,column=0,sticky='we')

            Button(self, text = 'Extract Selected Beads', command = self.ExtractBeads).grid(row=6,column=0,sticky='we')

            Button(self, text = 'Save beads', command = self.SaveSelectedBeads).grid(row=6,column=1,sticky='we')

            Button(self, text = 'Arithmetic average beads', command = self.BeadsArithmeticMean).grid(row=7,column=0,sticky='we')

            #test bead display canvas. May be removed.
            self.cnvImg = Canvas(self,  width = 200, height = 600, bg = 'white')
            self.cnvImg.grid(row = 0,column=3, rowspan=10,sticky=(N,E,W))
            Label(self, text = 'Bead Preview').grid(row=1,column=3)
            self.beadPrevNum = Entry(self, width = 5)
            self.beadPrevNum.grid(row=2,column = 3)
            self.beadPrevNum.insert(0,len(self.beadCoords))
            Button(self, text = "Preview Bead",command = self.PlotBeadPreview).grid(row=3,column = 3)
           



      def TestFunc1(self):
            """Loading raw beads photo from file"""
            self.beadImPaths = askopenfilenames(title = 'Load Beads Photo')
            print("Paths: ", self.beadImPaths)
            for i in self.beadImPaths :
                  print(i)
#             try:
#                  self.imgBeadsRaw = Image.open(self.beadImPath)
#                  print("Number of frames: ", self.imgBeadsRaw.n_frames)
#                  frameNumber = int( self.imgBeadsRaw.n_frames / 2)
#                  print("Frame number for output: ", frameNumber)
#                  # setting imgTmp to desired number
#                  self.imgBeadsRaw.seek(frameNumber)
#                  # preparing image for canvas from desired frame
#                  self.imgCnv = ImageTk.PhotoImage(self.imgBeadsRaw)
#            except:
#                  showerror("Error","Can't read file.")
#                  return
#            # replacing image on the canvas
#            self.cnv1.create_image(0, 0, image = self.imgCnv, anchor = NW)
#            # updating scrollers
#            self.cnv1.configure(scrollregion = self.cnv1.bbox('all')) 

      def SaveSelectedBeads(self):
            """Save selected beads as multi-page tiffs"""
            if hasattr(self, 'selectedBeads')  :
#                  txt_folder = self.folderPSFWgt.get()
#                  txt_prefix = self.filePrfxPSFWgt.get()
                  txt_folder = ''
                  txt_prefix = ''
                  if txt_prefix == '':
                        txt_prefix = "bead_"
                  if txt_folder == '':
                        dirId = -1
                  while True:
                        dirId += 1
                        print(dirId)
                        txt_folder = str(os.getcwd()) + "\\"+"bead_folder_"+str(dirId)
                        if not path.isdir(txt_folder):
                              print("creating dir")
                              os.mkdir(txt_folder)
                              break
                  for idx,bead in enumerate(self.selectedBeads):
                        fio.SaveTiffStack(bead,  txt_folder, txt_prefix+str(idx).zfill(2), self.colorDepth)
                        # the rest is test bead view print. May be removed later
                        self.imgBeadRaw = bead
                        # creating figure with matplotlib
                        fig, axs = plt.subplots(3, 1, sharex = False, figsize=(2,6))
                        axs[0].pcolormesh(self.imgBeadRaw[self.imgBeadRaw.shape[0] // 2,:,:],cmap=cm.jet)
                        axs[1].pcolormesh(self.imgBeadRaw[:,self.imgBeadRaw.shape[1] // 2,:],cmap=cm.jet)
                        axs[2].pcolormesh(self.imgBeadRaw[:,:,self.imgBeadRaw.shape[2] // 2],cmap=cm.jet)
                        # plt.show()
                        # Instead of plt.show creating Tkwidget from figure
                        self.figIMG_canvas_agg = FigureCanvasTkAgg(fig,self.cnvImg)
                        self.figIMG_canvas_agg.get_tk_widget().grid(row = 1,column=5, rowspan=10,sticky=(N,E,S,W))
#                        showinfo("Bead no. ", str(idx))
                  showinfo("Selected beads tiffs saved at saved at:", txt_folder)


      def PlotBeadPreview(self):
            """"Plots three bead in XYZ planes"""
            if len(self.beadCoords) <= 0:
                  showerror("PlotBeadPreview","Error. No beads selected")
            elif not hasattr(self,'selectedBeads'):
                  showerror("PlotBeadPreview","Error. Beads are not extracted.")
            else:
                  tmp = self.beadPrevNum.get()
                  if not tmp.isnumeric():
                        showerror("PlotBeadPreview", "Bad input")
                        self.beadPrevNum.delete(0,END)
                        self.beadPrevNum.insert(0,str(len(self.selectedBeads)-1))
                        return
                  else:
                        try:
                              self.imgBeadRaw = self.selectedBeads[int(tmp)]
                              # creating figure with matplotlib
                              fig, axs = plt.subplots(3, 1, sharex = False, figsize=(2,6))
                              axs[0].pcolormesh(self.imgBeadRaw[self.imgBeadRaw.shape[0] // 2,:,:],cmap=cm.jet)
                              axs[1].pcolormesh(self.imgBeadRaw[:,self.imgBeadRaw.shape[1] // 2,:],cmap=cm.jet)
                              axs[2].pcolormesh(self.imgBeadRaw[:,:,self.imgBeadRaw.shape[2] // 2],cmap=cm.jet)
                              # plt.show()
                              # Instead of plt.show creating Tkwidget from figure
                              self.figIMG_canvas_agg = FigureCanvasTkAgg(fig,self.cnvImg)
                              self.figIMG_canvas_agg.get_tk_widget().grid(row = 1,column=5, rowspan=10,sticky=(N,E,S,W))
                        except IndexError:
                              showerror("PlotBeadPreview", "Index out of range.")
                              self.beadPrevNum.delete(0,END)
                              self.beadPrevNum.insert(0,str(len(self.selectedBeads)-1))

      def BeadsArithmeticMean(self):
            if not hasattr(self,'selectedBeads'):
                  showerror("Error","Extract beads first.")
            else:
                  self.__avrageBead = sum(self.selectedBeads) / len(self.selectedBeads)
                  print("selectedBeads length: ",type(self.__avrageBead), self.__avrageBead.shape)

      
      def BeadMarkClickOld(self,event):
            """Append mouse event coordinates to global list."""
            cnv = event.widget
            self.xr,self.yr = cnv.canvasx(event.x),cnv.canvasy(event.y)
            self.beadMarks.append(cnv.create_rectangle(self.xr-self.sideHalf,self.yr-self.sideHalf,self.xr+self.sideHalf,self.yr+self.sideHalf, outline='chartreuse1',width = 2))
            self.beadCoords.append([self.xr,self.yr])

      def BeadMarkClick(self,event):
            """Append mouse event coordinates to global list."""
            cnv = event.widget
            self.xr,self.yr = cnv.canvasx(event.x),cnv.canvasy(event.y)
#            self.xr,self.yr = self.LocateFrameMAxIntensity2D()
            self.xr,self.yr = self.LocateFrameMAxIntensity3D()
            self.beadMarks.append(cnv.create_rectangle(self.xr-self.sideHalf,self.yr-self.sideHalf,self.xr+self.sideHalf,self.yr+self.sideHalf, outline='chartreuse1',width = 2))
            self.beadCoords.append([self.xr,self.yr])

      def LocateFrameMAxIntensity2D(self):
            """Locate point with maximum intensity in current 2d array.
                  In: array - np.array
                  Out: coords - list
            """
            d = self.sideHalf
            # dimension 0 - its z- plane
            # dimension 1 - y
            # dimension 2 - x
            xi =  self.xr
            yi =  self.yr
            bound3 = int(xi - d) 
            bound4 = int(xi + d)
            bound1 = int(yi - d)
            bound2 = int(yi + d)
#                  print("coords: ",bound1,bound2,bound3,bound4)
            sample = self.imgCnvArr[int( self.imgBeadsRaw.n_frames / 2),bound1:bound2,bound3:bound4]
            maximum = np.amax(sample)
            coords = np.unravel_index(np.argmax(sample, axis=None), sample.shape)
            #    print("LocateMaxIntensity: amax: ", maximum)
            print("LocateMaxIntensity: coords:", coords)
            return coords[2]+bound3,coords[1]+bound1

# TODO: 3D need additional testing. Maybe centering along z-axis also?
      def LocateFrameMAxIntensity3D(self):
            """Locate point with maximum intensity in current 3d array.
                  In: array - np.array
                  Out: coords - list
            """
            d = self.sideHalf
            # dimension 0 - its z- plane
            # dimension 1 - y
            # dimension 2 - x
            xi =  self.xr
            yi =  self.yr
            bound3 = int(xi - d) 
            bound4 = int(xi + d)
            bound1 = int(yi - d)
            bound2 = int(yi + d)
#                  print("coords: ",bound1,bound2,bound3,bound4)
            sample = self.imgCnvArr[:,bound1:bound2,bound3:bound4]
            maximum = np.amax(sample)
            coords = np.unravel_index(np.argmax(sample, axis=None), sample.shape)
            #    print("LocateMaxIntensity: amax: ", maximum)
            print("LocateMaxIntensity: coords:", coords)
            return coords[2]+bound3,coords[1]+bound1

      def PrintBeadsList(self):
            """Prints all bead coords."""
            print('Beads list: ', self.beadCoords)

      def RemoveLastMark(self):
            """Removes the last bead in the list"""
            self.beadCoords.pop()
            self.cnv1.delete(self.beadMarks[-1])
            self.beadMarks.pop()

      def ClearAllMarks(self):
            """Clears all bead marks"""
            self.beadCoords = []
            for sq in self.beadMarks:
                  self.cnv1.delete(sq)
            self.beadMarks = []

      def LoadBeadsPhoto(self):
            """Loading raw beads photo from file"""

            "Clear beads from old image"
            self.beadCoords.clear()
            self.beadMarks.clear()

            self.beadImPath = askopenfilename(title = 'Load Beads Photo')
            self.imgCnvArr, self.colorDepth = fio.ReadTiffStackFile(self.beadImPath)
            try:
                  self.imgBeadsRaw = Image.open(self.beadImPath)
                  print(self.imgBeadsRaw.info)
                  print("Number of frames: ", self.imgBeadsRaw.n_frames)
                  midFrameNumber = int( self.imgBeadsRaw.n_frames / 2)
                  print("Frame number for output: ", midFrameNumber)
                  # setting imgTmp to desired number
                  self.imgBeadsRaw.seek(midFrameNumber)
                  # preparing image for canvas from desired frame
                  self.imgCnv = ImageTk.PhotoImage(self.imgBeadsRaw)
            except:
                  showerror("Error","Can't read file.")
                  return
            # replacing image on the canvas
            self.cnv1.create_image(0, 0, image = self.imgCnv, anchor = NW)
            # updating scrollers
            self.cnv1.configure(scrollregion = self.cnv1.bbox('all'))  


      def ReturnSizeEntryContent(self,event):
            """Bead selection rectangle size change"""
            tmp = self.selectSizeEntry.get()
            if not tmp.isnumeric():
                  showerror("ReturnSizeEntryContent", "Bad input")
                  self.selectSizeEntry.delete(0,END)
                  self.selectSizeEntry.insert(0,self.sideHalf * 2)
                  return
            else:
                  self.sideHalf = int(float(tmp) / 2)
      def ExtractBeads(self):
            """Extracting bead stacks from picture set"""
            self.selectedBeads = []
            d = self.sideHalf
            print(self.imgCnvArr.shape)
            elem = np.ndarray([self.imgCnvArr.shape[0],d*2,d*2])
            for idx,i in enumerate(self.beadCoords):
                  bound3 = int(i[0] - d)
                  bound4 = int(i[0] + d)
                  bound1 = int(i[1] - d)
                  bound2 = int(i[1] + d)
#                  print("coords: ",bound1,bound2,bound3,bound4)
                  elem = self.imgCnvArr[:,bound1:bound2,bound3:bound4]
                  self.selectedBeads.append(elem)


if __name__ == '__main__':
      base1 = BeadExtraction()
      base1.mainloop()