import librosa
import soundfile as sf
import librosa.display
import numpy as np

class AudioParams:
    """
    Parameters used for the audio data
    """
    sr = 4000
    #duration = 0.2
    # Melspectrogram
    n_mels = 200
    fmin = 20
    fmax = 2000
    n_fft = 2048 // 8
    hop_length = 512 // 8
    
    
def compute_melspec(y, params, augment=True):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """

    #if(augment):
     #   noise_amp = 0.005 * np.random.uniform() * np.amax(y)
      #  y = y + noise_amp

    melspec = librosa.feature.melspectrogram(
        y=y, sr=params.sr, n_mels=params.n_mels, fmin=params.fmin, fmax=params.fmax, n_fft = params.n_fft, hop_length = params.hop_length
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)

    return melspec

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Converts a one channel array to a 3 channel one in [0, 255]
    Arguments:
        X {numpy array [H x W]} -- 2D array to convert
    Keyword Arguments:
        eps {float} -- To avoid dividing by 0 (default: {1e-6})
        mean {None or np array} -- Mean for normalization (default: {None})
        std {None or np array} -- Std for normalization (default: {None})
    Returns:
        numpy array [3 x H x W] -- RGB numpy array
    """
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

#y, sr = sf.read(i, always_2d=True)
#y = np.mean(y, 1)

def get_melspec_image(seq):
    X = seq.astype(float)
    #X = X/np.std(X)
    #X = X-np.mean(X)
    X = compute_melspec(X, AudioParams)
    X = mono_to_color(X)

    return X
