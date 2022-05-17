import glob, os, argparse, re, hashlib, time

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


#todo add silence + background data


class SpeechCommandsGoogle(Dataset):
    """Google Speech Command Dataset configured from Hello Edge"""

    def __init__(self, root_dir, train_test_val, val_perc, test_perc, words, sample_rate, batch_size, epochs, device, background_volume, background_frequency, silence_percentage, unknown_percentage, time_shift_ms, non_canonical_test = False, transform=None):
        self.sample_rate = sample_rate
        self.root_dir = root_dir
        self.transform = transform
        self.train_test_val = train_test_val
        self.val_perc = val_perc
        self.test_perc = test_perc
        self.words = words
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.non_canonical_test = non_canonical_test
        self.background_volume = background_volume
        self.background_frequency = background_frequency
        self.silence_percentage = silence_percentage
        self.unknown_percentage = unknown_percentage
        self.time_shift_ms = (time_shift_ms * sample_rate) / 1000

        self.noise = torch.distributions.bernoulli.Bernoulli(torch.tensor([self.background_frequency]))

        self.list_of_x = []
        self.list_of_labels = []
        self.list_of_y = []
        sub_dirs = [x[0].split('\\')[-1] for x in os.walk(root_dir)][1:]
        for cur_dir in sub_dirs:
            files_in_dir = glob.glob(root_dir + "\\" + cur_dir + "\\" + "*.wav")
            cur_dir = cur_dir.strip('_')
            for cur_f in files_in_dir:
                if cur_dir == "background_noise":
                    self.list_of_y.append(words.index('silence'))
                    self.list_of_labels.append('silence')
                elif which_set(cur_f, val_perc, test_perc) == train_test_val:
                    #if (cur_dir not in words) and (train_test_val != 'testing'):
                    if (cur_dir not in words) and ( not ((train_test_val == 'testing') and not non_canonical_test)):
                        self.list_of_y.append(words.index('unknown'))
                        self.list_of_labels.append('unknown')
                    else:
                        self.list_of_y.append(words.index(cur_dir))
                        self.list_of_labels.append(cur_dir)
                else:
                    continue

                waveform, sample_rate = torchaudio.load(cur_f)
                if sample_rate != self.sample_rate:
                    raise ValueError('Specified sample rate doesn\'t match sample rate in .wav file.')
                
                self.list_of_x.append(waveform)

        self.list_of_y = np.array(self.list_of_y)

        if (self.train_test_val == 'validation') or ((self.train_test_val == 'testing') and self.non_canonical_test):
            self.size = int(np.sum(np.unique(self.list_of_y, return_counts= True)[1][:10])/.8)
            self.deterministic_sampling = True

            self.relevant_lable_list = np.where(self.list_of_y < 10)[0]
        else:
            self.deterministic_sampling = False
            if self.train_test_val == 'testing':
                self.size = len(self.list_of_labels)
            else:
                self.size = int(self.batch_size * self.epochs)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if (self.train_test_val == 'testing') and (not self.non_canonical_test) and ('cough' not in self.root_dir):
            # usig canonical testing set which is already balanced     
            waveform = self.list_of_x[idx]
        else:
            # balance training and validation samples
            selector = idx/self.size

            if selector < self.silence_percentage:
                idx = np.random.choice(np.argwhere(self.list_of_y == 11)[:,0],1)
                waveform = torch.zeros(1,self.sample_rate)
            elif (selector >= self.silence_percentage) and (selector < (self.silence_percentage + self.unknown_percentage)):
                idx = np.random.choice(np.argwhere(self.list_of_y == 10)[:,0],1)
                waveform = self.list_of_x[idx.item()]
            else:
                if self.deterministic_sampling:
                    idx = self.relevant_lable_list[int(idx - self.size*.2)]
                else:
                    y_sel = int(np.floor(((selector - .2)/.8 * (len(self.words)-2))))
                    idx = np.random.choice(np.argwhere(self.list_of_y == y_sel)[:,0],1)
                waveform = self.list_of_x[idx.item()]

            # random time shift
            if self.time_shift_ms != 0:
                start_idx = int(np.random.choice(np.arange(0, self.time_shift_ms)))
                if (waveform.shape[1] - start_idx) >= self.sample_rate:
                    waveform = waveform[0,start_idx:(waveform.shape[1] + start_idx)].view(1,-1)
                elif (waveform.shape[1] - start_idx) < self.sample_rate:
                    pad_size = int((self.sample_rate - (waveform.shape[1] - start_idx))/2)
                    zero_waveform = torch.zeros((1,self.sample_rate))
                    zero_waveform[0,pad_size:(pad_size+(waveform.shape[1] - start_idx))] = waveform[0,start_idx:]
                    waveform = zero_waveform

            # sample noise
            if self.noise.sample() and self.train_test_val == 'training':
                noise_wave = self.list_of_x[np.random.choice(np.argwhere(self.list_of_y == 11)[:,0],1).item()]
                start_noise = int(np.random.choice(np.arange(0, noise_wave.shape[1] - (self.sample_rate+1))))
                noise_mul = noise_wave[0, start_noise:(start_noise+self.sample_rate)].view(1,-1) * self.background_volume
                waveform += noise_mul

        waveform = torch.clamp(waveform, min = -1., max = 1.)

        if waveform.shape[1] > self.sample_rate:
            # sample random 16000 from longer sequence
            start_idx = np.random.choice(np.arange(0,waveform.shape[1]-(self.sample_rate+1)))
            uniform_waveform = waveform[0,start_idx:(start_idx+self.sample_rate)].view(1,-1)
        elif waveform.shape[1] < self.sample_rate:
            # pad front and back with 0
            pad_size = int((self.sample_rate - waveform.shape[1])/2)
            uniform_waveform = torch.zeros((1,self.sample_rate))
            uniform_waveform[0,pad_size:(pad_size+waveform.shape[1])] =  waveform[0,:]
        else:
            uniform_waveform = waveform


        return uniform_waveform[0].t(), int(self.list_of_y[idx])
