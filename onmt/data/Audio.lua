local Audio = torch.class("Audio")

function Audio.getSpectrogram(file, windowSize, stride, sampleRate)
  if not audio then require 'audio' end

  local audioFile = audio.load(file)
  local spect = audio.spectrogram(audioFile, windowSize * sampleRate, 'hamming', stride * sampleRate)
  -- freq-by-frames tensor
  spect = spect:float()
  local mean = spect:mean()
  local std = spect:std()
  spect:add(-mean)
  spect:div(std)
  return spect:t() -- seqlength x feat size
end

function Audio.getMfcc(file,
                       sampleRate,
                       frameSize,
                       frameStride,
                       filterbankChannels,
                       cepstralCoefficients,
                       liftering,
                       lowerCutoffFrequency,
                       upperCutoffFrequency,
                       derivativeContextWindowSize,
                       melFloor)

  sampleRate = sampleRate or 16000
  frameSize = frameSize or 25
  frameStride = frameStride or 10
  filterbankChannels = filterbankChannels or 20
  cepstralCoefficients = cepstralCoefficients or 40
  liftering = liftering or 22
  lowerCutoffFrequency = lowerCutoffFrequency or 0
  upperCutoffFrequency = upperCutoffFrequency or sampleRate/2
  derivativeContextWindowSize = derivativeContextWindowSize or 9
  melFloor = melFloor or 0.0

  if not paths then paths = require ('paths') end
  if not speech then speech = require ('speech') end
  if not sndfile then sndfile = require ('sndfile') end

  local f_path = paths.thisfile(file)
  local file = sndfile.SndFile(f_path)
  local d = file:readFloat(file:info().frames):squeeze():double()
  local mfcc_f = speech.Mfcc{fs  = sampleRate,
                           tw  = frameSize,
                           ts  = frameStride,
                           M   = filterbankChannels,
                           N   = cepstralCoefficients,
                           L   = liftering,
                           R1  = lowerCutoffFrequency,
                           R2  = upperCutoffFrequency,
                           dev = derivativeContextWindowSize,
                           mel_floor = melFloor}
  local mfcc = mfcc_f(d)
  local mean = mfcc:mean()
  local std = mfcc:std()
  mfcc:add(-mean)
  mfcc:div(std)

  return mfcc
end

return Audio
