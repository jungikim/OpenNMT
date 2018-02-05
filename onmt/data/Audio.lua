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

return Audio