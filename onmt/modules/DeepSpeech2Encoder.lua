--[[ Deep Speech 2 implementation
--]]

local DeepSpeech2Encoder, parent = torch.class('onmt.DeepSpeech2Encoder', 'nn.Container')

local options = {
  {
    '-cnn_layers', 3,
    [[Number of convolutional layers of the encoder.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-rnn_layers', 2,
    [[Number of recurrent layers of the encoder.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  }
}

function DeepSpeech2Encoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end

function DeepSpeech2Encoder:__init(args, inputNetwork)

  parent.__init(self)

  self.inputNet = inputNetwork
  self.args = args
  self.args.numStates = 1

  local cnnInputDim = inputNetwork.inputSize


  local cnn_layers = self.args.cnn_layers

  local cnn = nn.Sequential()

  local cnnOutputDim

  if cnn_layers == 0 then
    cnn:add(nn.Identity())
    cnnOutputDim = 161
  else

  -- Baidu's DS2 CNN layer
  -- 3-layer 2D / 32, 32, 96 Channels / 41x11, 21x11, 21x11 filter dim / 2x2, 2x1, 2x1 stride
  cnn:add(nn.Transpose({1,2}):setNumInputDims(2)) --  batch x seqLength x dim -> batch x dim x seqlen
  cnn:add(nn.Unsqueeze(1,2)) -- add a single dimension for the input plane

  end

  --cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW = 1], [dH = 1], [padW = 0], [padH = 0], [groups = 1])

  -- nOutputPlane x oheight x owidth
  --owidth  = floor((width  + 2*padW - kW) / dW + 1)
  --oheight = floor((height + 2*padH - kH) / dH + 1)

  -- INPUT: Bx1x161x371

  if cnn_layers >= 1 then
  cnn:add(nn.SpatialConvolution(1, 32, 11, 41, 2, 2, 5, 20))
  cnn:add(nn.SpatialBatchNormalization(32, 1e-5, 0.1))
--  cnn:add(nn.ReLU6(true))
  cnn:add(nn.Clamp(0, 20))

  -- OUTPUT: [nOutputPlane] x [(161 + 40 - 41) / 2 + 1] x [(371 + 10 - 11) / 2 + 1] 
  --         Bx32x81x186

  cnnOutputDim = 32 * 81

  end


  if cnn_layers >= 2 then
  cnn:add(nn.SpatialConvolution(32, 32, 11, 21, 2, 1, 5, 0))
  cnn:add(nn.SpatialBatchNormalization(32, 1e-5, 0.1))
--  cnn:add(nn.ReLU6(true))
  cnn:add(nn.Clamp(0, 20))

  -- 16x32x61x43

  -- OUTPUT: [nOutputPlane] x [(81 - 21) / 1 + 1] x [(186 + 10 - 11) / 2 + 1] 
  --         Bx32x61x93

  cnnOutputDim = 32 * 61
  end

  if cnn_layers >= 3 then
  cnn:add(nn.SpatialConvolution(32, 96, 11, 21, 2, 1, 5, 0))
  cnn:add(nn.SpatialBatchNormalization(96, 1e-5, 0.1))
--  cnn:add(nn.ReLU6(true))
  cnn:add(nn.Clamp(0, 20))

  -- OUTPUT: [nOutputPlane] x [(61 - 21) / 1 + 1] x [(93 + 10 - 11) / 2 + 1] 
  --         Bx96x41x47

  cnnOutputDim = 96 * 41
  end

  if cnn_layers > 0 then
  cnn:add(nn.View(cnnOutputDim, -1):setNumInputDims(3)) -- batch x features x seqLength
  -- B x 3936 x seqLen
  cnn:add(nn.Transpose({2,3})) --  batch x seqLength x features
--  cnn:add(nn.Contiguous())
  -- B x 3936 x seqLen
  end

  require 'rnn'
  -- RNN layers
  local rnnUnit = nn.LSTM
  if self.args.rnn_type == 'GPU' then
    rnnUnit = nn.GRU
  end
  local rnn_layers = self.args.rnn_layers
  local rnn_size = self.args.rnn_size
  local dropout = self.args.dropout

--  print(self.args)
--  print(rnn_layers)
--  print(rnn_size)
--  print(dropout)
--  print(rnnUnit)

  local stepmodule = nn.Sequential() -- applied at each time-step
  for i=1, rnn_layers do
    local rnnModule
    if i==1 then
      rnnModule = rnnUnit(cnnOutputDim, rnn_size)
    else
      rnnModule = rnnUnit(rnn_size, rnn_size)
    end
    stepmodule:add(rnnModule)
    stepmodule:add(nn.Bottle(nn.BatchNormalization(rnn_size), 3))
    if dropout > 0 then
      stepmodule:add(nn.Dropout(dropout))
    end
  end
  local rnn = nn.Sequential()
  rnn:add(nn.Transpose({1,2})) --  batch x seqLength x features -> seqLength x batch x features
  rnn:add(nn.Contiguous())
  rnn:add(nn.Sequencer(stepmodule))
  rnn:add(nn.Transpose({1,2})) --  seqLength x batch x features -> batch x seqLength x features
  rnn:add(nn.Contiguous())

  self:add(cnn)
  self:add(rnn)

  self:resetPreallocation()
end


--[[ Return a new DeepSpeech2Encoder using the serialized data `pretrained`. ]]
function DeepSpeech2Encoder.load(pretrained)
  local self = torch.factory('onmt.DeepSpeech2Encoder')()
  parent.__init(self)

  self.modules = pretrained.modules
  self.args = pretrained.args
  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function DeepSpeech2Encoder:serialize()
  return {
    name = 'DeepSpeech2Encoder',
    modules = self.modules,
    args = self.args
  }
end


function DeepSpeech2Encoder:resetPreallocation()
  -- Prototype for preallocated state output gradients.
  self.gradStatesOutputProto = torch.Tensor()
  self._cnnOut = torch.Tensor()
  self._context = torch.Tensor()

end

function DeepSpeech2Encoder:forward(batch)

--  print('Input size: ' .. tostring(batch:getSourceInput():size()))
  -- B x 400 x 161
  self._cnnOut = self.modules[1]:forward(batch:getSourceInput())
--  print('CNN output size: ' .. tostring(self._cnnOut:size()))
  --B x 42 x 3936
  self._context = self.modules[2]:forward(self._cnnOut)
--  print('RNN output size: ' .. tostring(self._context:size()))
  --B x 42 x 500

  -- states: B x outputDim -- should be the RNN at the last step 
  -- context: B x seqLen x outputDim
  return nil, self._context
end

function DeepSpeech2Encoder:backward(batch, gradStatesOutput, gradContextOutput)
--  print('gradContextOutput: ' .. tostring(gradContextOutput:size())) -- 1x42x500
--  print('type(gradContextOutput): ' .. tostring(torch.type(gradContextOutput)))

--  local gcNaN_mask = gradContextOutput:ne(gradContextOutput)
--  gradContextOutput[gcNaN_mask] = 0
--  print('criterion NaN_mask: ' .. tostring(gcNaN_mask:sum()))

--  onmt.train.Optim.clipGradByNorm({gradContextOutput}, 5) -- self.max_grad_norm)

  local cnnGradOutput = self.modules[2]:backward(self._context, gradContextOutput)
--  print('rnnGradInput: ' .. tostring(rnnGradInput))

--  local cgNaN_mask = cnnGradOutput:ne(cnnGradOutput)
--  cnnGradOutput[cgNaN_mask] = 0
--  print('rnn NaN_mask: ' .. tostring(cgNaN_mask:sum()))

  local gradInput = self.modules[1]:backward(self._cnnOut, cnnGradOutput)
--  print('gradInput: ' .. tostring(gradInput))

--  local giNaN_mask = gradInput:ne(gradInput)
--  gradInput[giNaN_mask] = 0
--  print('cnn NaN_mask: ' .. tostring(giNaN_mask:sum()))

  return gradInput
end

