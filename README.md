# cognitiveProsthetic


import numpy as np
import mne as mne
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
# execfile('definitions.py')
# execfile('init.py')
# execfile('actions.py')
def init():
mnePath = mne.datasets.sample.data_path()
raw_fname = mnePath + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname)
baseline = (None, 0)
reject = dict(eog=150e-6) # grad=4000e-13, mag=4e-12, FROM BEGIN
events = mne.find_events(raw) # removed line stim_channel='STI 014' even though it
worked fine.
event_id = dict(aud_l=1, vis_l=3, aud_r=2, vis_r=4, unk=5, unkk=32) # I DONT GET THIS
YET
tmin = -0.2
tmax = 0.5
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, stim=False,
exclude=raw.info['bads'] + ['MEG 2443', 'EEG 053']) #
changed MEG from True to False
def epochGetter():
mnePath = mne.datasets.sample.data_path()
raw_fname = mnePath + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname)
baseline = (None, 0)
reject = dict(eog=150e-6) # grad=4000e-13, mag=4e-12, FROM BEGIN
events = mne.find_events(raw) # removed line stim_channel='STI 014' even though it
worked fine.
event_id = dict(aud_l=1, vis_l=3, aud_r=2, vis_r=4, unk=5, unkk=32) # I DONT GET THIS
YET
tmin = -0.2
tmax = 0.5
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, stim=False,
exclude=raw.info['bads'] + ['MEG 2443', 'EEG 053']) #
changed MEG from True to False
return mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
baseline=baseline, preload=False), raw
def resh3to2d(inData): # NOT VERY USEFUL
ebe = np.reshape(inData,
(inData.shape[0], inData.shape[1] * inData.shape[2]))
return ebe
def epochLinearize(inData): # NOT VERY USEFUL
ebe = np.reshape(inData,
(inData.shape[1], inData.shape[0] * inData.shape[2]))
return ebe
def zero_to_nan(values): # This is a means of plotting the stim channel.
"""Replace every 0 with 'nan' and return a copy."""
dashape = values.shape
values = values.flatten()
values = np.array([float('nan') if x == 0 else x for x in values])
return values.reshape(dashape)
def selectFreq(inp, *args): # this shall return the morlet screened for frequencies. All
channels.

# ARGS can be either one or two values. They will be dumped into slice syntax.
# It is designed to take a chunk of the morlet array
# In later development, we will perform the morlet transform AFTER the channels are
specified.
# THE INPUT WILL ARRIVE IN A BOX
#
if len(args) == 1:
return inp[:, args[0]]
if len(args) == 2:
return inp[:, args[0]:args[1]]
if len(args) == 0:
print('You are returning the entire set of frequencies')
return inp[:]
else:
print('Two or fewer args are necessary. Returning zilch.')
return 0
def moment(inp, idx):
return inp.transpose()[idx]
# plt.matshow(zero_to_nan(getBox( moment ( selectFreq(tmor, 2 ), 201 ))))
def getBox(inp, v=False): #TAKES A MORLET-CREATED PHASE ARRAY OF ONE INSTANT
length = len(inp)
ee = zero_to_nan(np.zeros((len(inp) - 1, len(inp) - 1)))
e = np.array([inp[w:] for w in np.arange(1, len(inp) - 1)])
for ii in np.arange(len(inp) - 1): # e[ii] is is a tier of the ziggaraut
# e[ii] = [abs(lilmor[ii] - ee) for ee in e]
ee[ii] = np.append(np.abs(np.full_like(inp[ii + 1:], inp[ii]) - inp[ii + 1:]),
np.zeros_like(inp[:ii]))
if v:
print(ee.shape, " subtraction matrixes have been created.")
return zero_to_nan(ee)
def getBoxLEAN(inp, v=False): #TAKES A MORLET-CREATED PHASE ARRAY OF ONE INSTANT
r= [] #TEMPORARILY YOU ARE CHANGING THE TUPLE FROM LOC, CLOSENSS TO HERE-LOC-CLOSENESS
t = []
for ii in np.arange(len(inp) - 1): # WE WILL RETURN THE TWO BEST RATIOS
therest = inp[ii + 1:]
atIDXrepeated = np.full_like(inp[ii + 1:], inp[ii])
diffs = np.abs(atIDXrepeated - therest)
closeAddress = np.argmin(diffs) + ii
closeness =np.min(diffs)
if closeness < .0001:
r.append([closeAddress + 1, ii, .01])
else:
r.append([closeAddress + 1, ii, 1])
#print('supposedly, the ', closeAddress + ii, " is closest to ", ii, "with a
closeness of: ", closeness, "\n do you believe it?" )
canweSort = np.array(r)
zipp = zip(canweSort[:, 1], canweSort[:, 0])
trynsort = list(zipp)
sorteed = np.array(trynsort)
return np.sort(sorteed, axis=0)[:4]
def appendEventIdManyTimes(inp):
oneEpoch = inp.get_data()[0]
shape = oneEpoch.shape
return np.reshape(np.append(oneEpoch, np.full_like(inp.times, inp.events[-1][-1])),
(shape[0] + 1, shape[1]))

def splitDataByTypeFromRaw(raw):
eegs = mne.pick_types(raw.info, meg=False, eeg=True, exclude=raw.info['bads'] + ['MEG
2443', 'EEG 053'])
stims = mne.pick_types(raw.info, meg=False, stim=True, exclude=raw.info['bads'] + ['MEG
2443', 'EEG 053'])
eogs = mne.pick_types(raw.info, meg=False, eog=True, exclude=raw.info['bads'] + ['MEG
2443', 'EEG 053'])
edat = raw.get_data(eegs)
sdat = raw.get_data(stims)
odat = raw.get_data(eogs)
return (edat, sdat, odat)
# Can I get a BOXES that is filtered based on event_id
def epochsByEvent(epochs, event_id):
dat = epochs[event_id].get_data()
# ebe = np.reshape(dat, (
# dat.shape[1], dat.shape[0] * dat.shape[2]))
# return ebe
return dat
def getManyBoxes(inp, listFreq=[7]):
fmor = mne.time_frequency.tfr_array_morlet(inp, sfreq=raw.info['sfreq'],
freqs=listFreq, n_cycles=3,
output='phase')
f7 = selectFreq(fmor[0],
0) # EXPLICITLY CALLS ONLY THE FIRST FIRST MEMBER OF THE LIST, BE
SURE TO FIX THIS
# ------------THIS BELOW TAKES A LONG TIME, ONLY USES ONE CORE, SHOULD BE
MULTIPROCESSED OR SMTH
return np.array([getBoxLEAN(moment(f7, X)) for X in np.arange(f7.shape[-1])])
def tier1(): #These are the tasks which formed all of the data for the first half of the
paper. This is not truly a function, just a means of
#preventing the code from being ran without having to delete it.
epochs, raw = epochGetter()
vr = epochsByEvent(epochs, 'vis_r')
vl = epochsByEvent(epochs, 'vis_l')
vLf7mor = mne.time_frequency.tfr_array_morlet([vl], sfreq=raw.info['sfreq'],
freqs=[7], n_cycles=3, output='phase')
kmeans = getkmeans(vLf7mor[0,:,0,:].transpose())
N = kmeans.labels_.shape[0]
x = np.arange(N)
rr = np.random.RandomState(5)
y = np.ma.array(rr.random_sample(N) / 100)
c = kmeans.labels_
#I clipped out the multiplot language
# fig, a1 = plt.subplots()
# a2 = a1.twinx()
masks = []
colorz = ['red','orange','yellow','green','blue','indigo', 'violet', 'pink']
fig, a1 = plt.subplots()
for j in np.arange(8):
masks.append(np.ma.masked_where(c != j, y))
a1.scatter(x, masks[j], c=colorz[j], alpha=.55)
a1.plot(raw.get_data()[0][0])
a2 = a1.twinx()
for xc in mne.find_events(raw)[:, 0]:
a2.axvline(x=xc)

plt.show()
#BOXES
boxes = getManyBoxes(vl)
boxes = np.array([x[~np.isnan(x)] for x in boxes])
tempdat = appendEventIdManyTimes(epochs[3])
evoked = epochs['aud_l'].average()
evoked.plot()
def offsetTimes(raw):
eventTimes = mne.find_events(raw)
newtimes = eventTimes.transpose()[0] - eventTimes.transpose()[0][0]
eventTimes.transpose()[0] = newtimes #[:,1]
return eventTimes
eventID = ('aud_l', 'vis_l', 'aud_r', 'vis_r', 'unk', 'unkk')
epochs, raw = epochGetter()
ebeMap = map(epochsByEvent,
np.full(len(eventID), epochs), #these will be zipped together
eventID)
listOfEpochsbyEvent = list(ebeMap)
rawarray = mne.io.RawArray(raw.get_data(), raw.info, first_samp=0, verbose=None)
rawarray.add_events(offsetTimes(raw))
#add events, the offset version.
stimchannum = mne.pick_types(rawarray.info, stim=True, meg=False)[6]
stch = rawarray.get_data()[stimchannum]
stchIndices = np.where(stch != 0)[0]
stchVals = stch[stch!=0]
colorMap = {0:'g', 1:'b', 2:'r', 3: 'c' , 4: 'm' , 5: 'y', 7: 'xkcd:sky blue' , 32:
'xkcd:pink' ,35: 'xkcd:magenta' , 36: 'xkcd:sea green' }
##This works very very well
for k in np.arange(stchVals.shape[0]):
plt.axvline(stchIndices[k], color=colorMap[stchVals[k]], linewidth=.25)
#infor for the dataset. You also downloaded this.
#https://s3.amazonaws.com/academia.edu.documents/34924093/Viola09.pdf?
AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1556472163&Signature=VlqsEkpdRY4nkoMFkdY5DOVUq8

g%3D&response-content-disposition=inline%3B%20filename%3DSemi-
automatic_identification_of_indepen.pdf

#CURRENTLY YOU ARE WORKING ON THE DUPLICATE OF GETBOX, THE ONE WHICH WILL RETURN KEY
FEATURES AND NOT A BOX
manyBOXXES = getManyBoxes(listOfEpochsbyEvent[0])
resh3to2d(manyBOXXES) #flattened
#kmm, vqq = getkmeans(resh3to2d(manyBOXXES))
classifiedBoxesFlat = []
for k in np.arange(4):
temp = [epochLinearize(listOfEpochsbyEvent[k])]
temp2 = resh3to2d(getManyBoxes(temp))
classifiedBoxesFlat.append(temp2)
trainingSet = []
for K in listOfEpochsbyEvent[:4]:
trainingSet.append(K[:20])
trainingSet = np.concatenate(np.array(trainingSet), axis=0)
trainingbox = getManyBoxes([epochLinearize(trainingSet)])
#TEMPORARILY, trainingbox is N , 4 , 2 in shape, because you are returning 4 phase
locations.
#so you must call resh3to2d(trainingbox), when you return one phase loc, you wont use the
RESH call
trainedCodeBook, trainedDistortion = kmeans(resh3to2d(trainingbox), 100)
#codebook, distortion = getkmeans(resh3to2d(trainingbox))
#can I reverse it?
selfTestResult = kmeans(resh3to2d(trainingbox), trainedCodeBook)
#codebooktest, distortion = kmeans(resh3to2d(classifiedBoxesFlat[0]), trainedCodeBook)
labelsofSelf, distofSelf = vq(resh3to2d(trainingbox), trainedCodeBook)

labelsof0, distof0 = vq(classifiedBoxesFlat[0], trainedCodeBook)
labelsof1, distof1 = vq(classifiedBoxesFlat[1], trainedCodeBook)
labelsof2, distof2 = vq(classifiedBoxesFlat[2], trainedCodeBook)
labelsof3, distof3 = vq(classifiedBoxesFlat[3], trainedCodeBook)