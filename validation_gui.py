# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:41:19 2021

@author: Sebastian Menze
"""

import soundfile as sf
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt
import time
import os
import glob
import librosa

#%% user input

# the folder conatining the audio data 
audio_folder=r'I:\postdoc_krill\pam\2016_aural'

# the code for date and time information in the files
timecode='aural_%Y_%m_%d_%H_%M_%S.wav'

# the pathto the csv file containg the detection timestamps
detectioncsv=r"C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\specgram_corr\automaticdetections_dcall_full_2016.csv"

# the space left and right of the detection in the plor (in seconds)
offset_sec=10 

# the frequency limits of the spectrogram
f_lim=[15,150]

# the spectrogram resolution
fft_size=2**14

# if yo have already labeled parts of the detections, specify the path to your old csv file here
# old_csv='detection_validations.csv'


#%% load audio

# audio_folder=r'I:\postdoc_krill\pam\2016_aural'
audiopaths=glob.glob(audio_folder+'\*.wav')
audiostarttimes=[]

audioduration=[]

for audiopath in audiopaths:         
    # print(audiopath)    
    audiostarttimes.append( dt.datetime.strptime( audiopath.split('\\')[-1], timecode ) )
    # x,fs=sf.read(audiopath,dtype='int16')
    # duration=len(x)/fs
    audioduration.append(    librosa.get_duration(filename=audiopath) )

audiostarttimes=pd.Series(audiostarttimes)     
audioduration=pd.Series(audioduration)     
audioendtimes= audiostarttimes + pd.to_timedelta(audioduration,'s')

#%% load detections

# detectioncsv=r"C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\specgram_corr\automaticdetections_dcall_full_2016.csv"
df=pd.read_csv(detectioncsv)    
detections=  pd.to_datetime(df.iloc[:,1])

# or something like:
    
# detections=[]
# csv_names=glob.glob(folder)
# for path in csv_names:
#     df=pd.read_csv(path)
#     ix=df['Label']=='BW_D_call'
#     detections.append(df['Timestamp'][ix])
# detections = pd.concat(detections,ignore_index=True)
# detections=pd.to_datetime(detections)   

# assign audiopath to each detection
detections_af=[]
for detec in detections:
    ix = np.where( (detec>audiostarttimes) & (detec<audioendtimes)  )[0][0]
    detections_af.append(  audiopaths[ix]  )

#%% set labels and load labels from previous session

labels=np.ones(np.shape(detections))*np.nan

if ('old_csv' in locals()):
    a=pd.read_csv(old_csv, index_col=0)
    labels=a.iloc[:,1].values
        
#%% clicking gui

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
# from PyQt5.QtWidgets import QShortcut
# from PyQt5.QtGui import QKeySequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg ):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.canvas =  MplCanvas(self, width=5, height=4, dpi=150)   
       ##############
        self.audiopath_old='jkbfaa'
        
        self.detections_af=detections_af
        self.detections=detections
        self.labels=labels
        self.ixnan=np.where(pd.Series(self.labels).isna())[0]
        self.ii=0
        self.ix=self.ixnan[self.ii]
        
        def plot_detection():
            self.audiopath=self.detections_af[self.ix]
            if self.ii>0:
                   self.audiopath_old=self.detections_af[self.ixnan[self.ii-1]]
         
            if not self.audiopath==self.audiopath_old:     
                x, fs = sf.read(self.audiopath,dtype='int16')   
                audiostarttime= dt.datetime.strptime( self.audiopath.split('\\')[-1],timecode) 
        
                fft_size=2**14
                self.f, self.t, self.Sxx = signal.spectrogram(x, fs, window='hamming',nperseg=fft_size,noverlap=fft_size*0.9)
                self.rectime=audiostarttime + pd.to_timedelta( self.t ,'s')
                        
            detection_time=self.detections[self.ix]
            # print(detection_time)
            # print(self.rectime)
     
            self.canvas.fig.clf() 
            self.canvas.axes = self.canvas.fig.add_subplot(111)
            
            t1=detection_time - pd.Timedelta(offset_sec,'s')
            t2=detection_time + pd.Timedelta(offset_sec,'s')          
            ix_t=np.where( (self.rectime>=t1) & (self.rectime<=t2) )[0]
            ix_f=np.where((self.f>=f_lim[0]) & (self.f<=f_lim[1]))[0]
            spectrog =10*np.log10( self.Sxx[ ix_f[0]:ix_f[-1],ix_t[0]:ix_t[-1] ] )
            
            self.canvas.axes.imshow(spectrog ,cmap='inferno',aspect='auto',origin = 'lower',extent = [self.t[ix_t[0]],self.t[ix_t[-1]] , f_lim[0], f_lim[-1]])
            self.canvas.axes.set_ylabel('Frequency [Hz]')
            self.canvas.axes.set_xlabel('Time [sec]')
              
            td=np.argmin(np.abs(self.rectime-detection_time))
            self.canvas.axes.plot([self.t[td],self.t[td]], f_lim , '--k')
            # self.canvas.axes.arrow(self.t[td], 100, 0,-20, length_includes_head=True,
            #       width=1)
            
            targetname=detection_time.strftime('%Y_%m_%d_%H_%M_%S')
            self.canvas.axes.set_title(str(self.ix)+' of ' + str(len(self.detections)) +': ' + targetname)
            self.canvas.fig.tight_layout()
            self.canvas.draw()            
        

 ######## layout
        outer_layout = QtWidgets.QVBoxLayout()
        
        top_layout = QtWidgets.QHBoxLayout()       
        
        button_yes=QtWidgets.QPushButton('Yes')
        def yes_func():                
            plot_detection()
            self.labels[self.ix]=1
            self.ii=self.ii+1  
            self.ix=self.ixnan[self.ii]

            if self.ii>=len(self.ixnan):
                self.ii=len(self.ixnan)-1
                print('THE END')
            # export data
            annot=pd.concat([self.detections,pd.Series(self.labels)],axis=1,ignore_index=True)
            annot.to_csv('detection_validations.csv')
            
        button_yes.clicked.connect(yes_func)     
        button_yes.setStyleSheet("background-color: green")
        top_layout.addWidget(button_yes)

    
        button_no=QtWidgets.QPushButton('No')
        def no_func():                
            plot_detection()
            self.labels[self.ix]=0
            self.ii=self.ii+1  
            self.ix=self.ixnan[self.ii]
            if self.ii>=len(self.ixnan):
                self.ii=len(self.ixnan)-1
                print('THE END')
            # export data
            annot=pd.concat([self.detections,pd.Series(self.labels)],axis=1,ignore_index=True)
            annot.to_csv('detection_validations.csv')    
        button_no.clicked.connect(no_func)    
        button_no.setStyleSheet("background-color: red")
        top_layout.addWidget(button_no)
        
        button_previous=QtWidgets.QPushButton('<--Previous')
        def previous_func():    
            self.ii=self.ii-1  
            self.ix=self.ixnan[self.ii]
            if self.ix<0:
                self.ix=0
            plot_detection()
            self.labels[self.ix]=0
            print(self.labels)
            self.ix=self.ix+1           
        button_previous.clicked.connect(previous_func)    
        top_layout.addWidget(button_previous)
        
        button_next=QtWidgets.QPushButton('Next-->')
        def next_func():    
            self.ii=self.ii+1  
            self.ix=self.ixnan[self.ii]
            if self.ix<0:
                self.ix=0
            plot_detection()
            self.labels[self.ix]=0
            print(self.labels)
            self.ix=self.ix+1           
        button_next.clicked.connect(next_func)    
        top_layout.addWidget(button_next)
        
        
        # top_layout.addWidget(button_no)
        # top_layout.addWidget(button_previous)
        button_quit=QtWidgets.QPushButton('Quit')
        button_quit.clicked.connect(QtWidgets.QApplication.instance().quit)
        top_layout.addWidget(button_quit)


        # combine layouts together
        
        plot_layout = QtWidgets.QVBoxLayout()
        toolbar = NavigationToolbar( self.canvas, self)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)
        
        outer_layout.addLayout(top_layout)
        outer_layout.addLayout(plot_layout)
        
        # self.setLayout(outer_layout)
        
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(outer_layout)
        self.setCentralWidget(widget)
        
        # #### hotkeys
        self.msgSc1 = QtWidgets.QShortcut(QtCore.Qt.Key_Right, self)
        self.msgSc1.activated.connect(next_func)
        self.msgSc2 = QtWidgets.QShortcut(QtCore.Qt.Key_Left, self)
        self.msgSc2.activated.connect(previous_func)        
 
      
        plot_detection()
        self.show()


app = QtWidgets.QApplication(sys.argv)
app.setApplicationName("validate")

w = MainWindow()
sys.exit(app.exec_())


#%% loop that pots all detections

# audiopath_old='a'

# for ix in range(len(detections)):     
    
#     audiopath=detections_af[ix]
#     if not audiopath==audiopath_old:     
#         x, fs = sf.read(audiopath,dtype='int16')   
#         audiostarttime= dt.datetime.strptime( audiopath.split('\\')[-1], 'aural_%Y_%m_%d_%H_%M_%S.wav' ) 

#         fft_size=2**14
#         f, t, Sxx = signal.spectrogram(x, fs, window='hamming',nperseg=fft_size,noverlap=fft_size*0.9)
#         rectime=audiostarttime + pd.to_timedelta( t ,'s')
                
#     detection_time=detections[ix]
#     fig=plt.figure(num=1)      
#     plt.clf()
#     fig.set_size_inches(5, 5)
    
#     offset_sec=10
#     t1=detection_time - pd.Timedelta(offset_sec,'s')
#     t2=detection_time + pd.Timedelta(offset_sec,'s')          
#     ix_t=np.where( (rectime>=t1) & (rectime<=t2) )[0]
#     f_lim=[15,150]
#     ix_f=np.where((f>=f_lim[0]) & (f<=f_lim[1]))[0]
#     spectrog =10*np.log10( Sxx[ ix_f[0]:ix_f[-1],ix_t[0]:ix_t[-1] ] )
    
#     plt.imshow(spectrog ,cmap='inferno',aspect='auto',origin = 'lower',extent = [t[ix_t[0]],t[ix_t[-1]] , f_lim[0], f_lim[-1]])
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.yscale('linear')
      
#     td=np.argmin(np.abs(rectime-detection_time))
#     plt.arrow(t[td], 100, 0,-20, length_includes_head=True,
#           width=1)
        
#     plt.title(detection_time)   
#     plt.tight_layout()
#     plt.ion()
#     fig.canvas.draw()
#     plt.show()

#     input("Press Enter to continue...")
    
    
#     targetname=pd.Timestamp(detection_time).strftime('BW_dcall_time_%Y_%m_%d_%H_%M_%S')
#     print(targetname)
#     audiopath_old=audiopath
#     plt.savefig( image_folder+'\\'+ targetname+'.jpg',dpi=150 )
               
