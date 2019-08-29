import mne as mne

pathn = mne.datasets.sample.data_path()
pathx = pathn + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(pathx)

mock = mne.realtime.MockLSLStream()

sample_audvis_filt-0-40_raw.fif